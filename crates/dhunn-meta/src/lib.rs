//! Track metadata. For now: reading embedded tags (ID3, Vorbis comments, MP4 atoms, RIFF
//! INFO, …) via lofty, and parsing artist credits. MusicBrainz/ListenBrainz come later.

use std::path::Path;

use dhunn_core::{ArtistCredit, ArtistRole};
use lofty::prelude::*;
use lofty::tag::{ItemKey, Tag};

/// What the tags say about a file. Every field is optional: plenty of files have no tags.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Tags {
    pub title: Option<String>,
    pub album: Option<String>,
    pub year: Option<i32>,
    pub mb_recording_id: Option<String>,
    pub artists: Vec<ArtistCredit>,
}

/// Reads tags from a file. Unreadable or missing tags give `Tags::default()` — a file with
/// bad tags is still perfectly good audio.
pub fn read_tags(path: &Path) -> Tags {
    let Ok(file) = lofty::read_from_path(path) else {
        return Tags::default();
    };
    let Some(tag) = file.primary_tag().or_else(|| file.first_tag()) else {
        return Tags::default();
    };
    tags_from(tag)
}

fn clean(s: Option<&str>) -> Option<String> {
    s.map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_owned)
}

fn tags_from(tag: &Tag) -> Tags {
    let title = clean(tag.get_string(ItemKey::TrackTitle));
    let mb_artist = clean(tag.get_string(ItemKey::MusicBrainzArtistId));

    // Prefer an explicit multi-value ARTISTS list; otherwise parse the display string(s).
    let listed: Vec<&str> = tag.get_strings(ItemKey::TrackArtists).collect();
    let mut artists = if !listed.is_empty() {
        listed
            .iter()
            .filter_map(|n| clean(Some(n)))
            .enumerate()
            .map(|(i, name)| {
                credit(
                    name,
                    if i == 0 {
                        ArtistRole::Primary
                    } else {
                        ArtistRole::Featured
                    },
                )
            })
            .collect()
    } else {
        let display: Vec<&str> = tag.get_strings(ItemKey::TrackArtist).collect();
        let display = if display.is_empty() {
            tag.get_strings(ItemKey::AlbumArtist).collect()
        } else {
            display
        };
        display
            .iter()
            .flat_map(|s| parse_artist_credits(s))
            .collect::<Vec<_>>()
    };
    if let Some(t) = &title {
        for name in featured_in_title(t) {
            artists.push(credit(name, ArtistRole::Featured));
        }
    }
    dedupe(&mut artists);
    if let (Some(first), Some(id)) = (artists.first_mut(), mb_artist) {
        first.mb_artist_id = Some(id);
    }

    let year = [
        ItemKey::Year,
        ItemKey::RecordingDate,
        ItemKey::ReleaseDate,
        ItemKey::OriginalReleaseDate,
    ]
    .into_iter()
    .find_map(|k| tag.get_string(k).and_then(parse_year));

    Tags {
        title,
        album: clean(tag.get_string(ItemKey::AlbumTitle)),
        year,
        mb_recording_id: clean(tag.get_string(ItemKey::MusicBrainzRecordingId)),
        artists,
    }
}

fn credit(name: String, role: ArtistRole) -> ArtistCredit {
    ArtistCredit {
        name,
        role,
        mb_artist_id: None,
    }
}

fn dedupe(artists: &mut Vec<ArtistCredit>) {
    let mut seen = std::collections::HashSet::new();
    artists.retain(|a| seen.insert(dhunn_core::normalize_name(&a.name)));
}

/// Markers that introduce featured artists, matched case-insensitively.
const FEAT_MARKERS: &[&str] = &[
    " featuring ",
    " feat. ",
    " feat ",
    " ft. ",
    " ft ",
    " with ",
];

/// Separators between several primary artists. `&`, `,` and `and` are deliberately *not*
/// split: they are part of too many single names ("Simon & Garfunkel", "Earth, Wind & Fire").
const PRIMARY_SEPARATORS: &[&str] = &[";", " / ", "\u{0}"];

/// Separators inside a featured list ("feat. A, B & C").
const FEATURED_SEPARATORS: &[&str] = &[",", " & ", " and ", ";", " / "];

fn split_all<'a>(s: &'a str, seps: &[&str]) -> Vec<&'a str> {
    let mut parts = vec![s];
    for sep in seps {
        parts = parts.into_iter().flat_map(|p| p.split(sep)).collect();
    }
    parts
        .into_iter()
        .map(str::trim)
        .filter(|p| !p.is_empty())
        .collect()
}

/// Finds the earliest featuring marker (also in "(feat. …)" / "[ft. …]" form). Returns the
/// text before it and the featured part with brackets removed.
fn split_featuring(s: &str) -> Option<(&str, &str)> {
    let lower = s.to_lowercase();
    // Byte offsets in `lower` equal those in `s` only for ASCII; bail out otherwise.
    if lower.len() != s.len() {
        return None;
    }
    let mut best: Option<(usize, usize)> = None;
    for m in FEAT_MARKERS {
        for open in ["(", "[", ""] {
            let needle = if open.is_empty() {
                m.to_string()
            } else {
                format!(" {open}{}", m.trim_start())
            };
            if let Some(i) = lower.find(&needle)
                && best.is_none_or(|(b, _)| i < b)
            {
                best = Some((i, i + needle.len()));
            }
        }
    }
    let (start, end) = best?;
    let feat = s[end..].trim().trim_end_matches([')', ']']).trim();
    Some((&s[..start], feat))
}

/// Parses a display artist string such as `"A feat. B & C"` or `"A; B"`.
pub fn parse_artist_credits(s: &str) -> Vec<ArtistCredit> {
    let (main, featured) = match split_featuring(s) {
        Some((m, f)) => (m, Some(f)),
        None => (s, None),
    };
    let mut out: Vec<ArtistCredit> = split_all(main, PRIMARY_SEPARATORS)
        .into_iter()
        .map(|n| credit(n.to_owned(), ArtistRole::Primary))
        .collect();
    if let Some(f) = featured {
        out.extend(
            split_all(f, FEATURED_SEPARATORS)
                .into_iter()
                .map(|n| credit(n.to_owned(), ArtistRole::Featured)),
        );
    }
    out
}

/// Featured artists named in a title like `"Song (feat. A & B)"`. Only bracketed forms
/// count — a bare " with " inside a title is usually just a word.
pub fn featured_in_title(title: &str) -> Vec<String> {
    let lower = title.to_lowercase();
    if lower.len() != title.len() {
        return Vec::new();
    }
    for open in ['(', '['] {
        for m in ["feat. ", "feat ", "ft. ", "ft ", "featuring "] {
            let needle = format!("{open}{m}");
            if let Some(i) = lower.find(&needle) {
                let rest = &title[i + needle.len()..];
                let close = if open == '(' { ')' } else { ']' };
                let inner = rest.split(close).next().unwrap_or("");
                return split_all(inner, FEATURED_SEPARATORS)
                    .into_iter()
                    .map(str::to_owned)
                    .collect();
            }
        }
    }
    Vec::new()
}

/// Splits an untagged file name like `"03 - Artist - Title"` or `"Artist - Title"` into
/// `(artist, title)`, dropping a leading track number. `None` if there is no " - ".
pub fn parse_file_name(stem: &str) -> Option<(String, String)> {
    let mut s = stem.trim();
    // Leading track number: "03 ", "03. ", "1-03 - ", "03 - ".
    let digits = s
        .find(|c: char| !(c.is_ascii_digit() || c == '-'))
        .unwrap_or(0);
    if digits > 0 && digits <= 5 {
        let rest = s[digits..].trim_start_matches(['.', ' ', '_']).trim_start();
        let rest = rest.strip_prefix("- ").unwrap_or(rest);
        if !rest.is_empty() {
            s = rest;
        }
    }
    let (artist, title) = s.split_once(" - ")?;
    let (artist, title) = (artist.trim(), title.trim());
    (!artist.is_empty() && !title.is_empty()).then(|| (artist.to_owned(), title.to_owned()))
}

/// First plausible 4-digit year in a date string ("1970", "1970-05-08", "2003/01/01").
pub fn parse_year(s: &str) -> Option<i32> {
    let b = s.as_bytes();
    (0..b.len().saturating_sub(3)).find_map(|i| {
        let w = &b[i..i + 4];
        let digits = w.iter().all(u8::is_ascii_digit);
        let bounded = (i == 0 || !b[i - 1].is_ascii_digit())
            && b.get(i + 4).is_none_or(|c| !c.is_ascii_digit());
        if digits && bounded {
            let y: i32 = std::str::from_utf8(w).ok()?.parse().ok()?;
            (1000..=2999).contains(&y).then_some(y)
        } else {
            None
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn names(c: &[ArtistCredit]) -> Vec<(&str, ArtistRole)> {
        c.iter().map(|a| (a.name.as_str(), a.role)).collect()
    }

    use ArtistRole::{Featured as F, Primary as P};

    #[test]
    fn parses_artist_credits() {
        assert_eq!(
            names(&parse_artist_credits("Simon & Garfunkel")),
            vec![("Simon & Garfunkel", P)]
        );
        assert_eq!(
            names(&parse_artist_credits("Earth, Wind & Fire")),
            vec![("Earth, Wind & Fire", P)]
        );
        assert_eq!(
            names(&parse_artist_credits(
                "Daft Punk feat. Pharrell Williams & Nile Rodgers"
            )),
            vec![
                ("Daft Punk", P),
                ("Pharrell Williams", F),
                ("Nile Rodgers", F)
            ]
        );
        assert_eq!(
            names(&parse_artist_credits("A FT. B")),
            vec![("A", P), ("B", F)]
        );
        assert_eq!(
            names(&parse_artist_credits("A (feat. B, C)")),
            vec![("A", P), ("B", F), ("C", F)]
        );
        assert_eq!(
            names(&parse_artist_credits("A; B")),
            vec![("A", P), ("B", P)]
        );
        assert_eq!(names(&parse_artist_credits("AC/DC")), vec![("AC/DC", P)]);
        assert!(parse_artist_credits("  ").is_empty());
    }

    #[test]
    fn featured_from_title() {
        assert_eq!(featured_in_title("Song (feat. A & B)"), vec!["A", "B"]);
        assert_eq!(featured_in_title("Song [ft. C]"), vec!["C"]);
        assert!(featured_in_title("Stand By Me with Strings").is_empty());
    }

    #[test]
    fn file_names() {
        let p = |s| parse_file_name(s);
        assert_eq!(
            p("The Beatles - Let It Be"),
            Some(("The Beatles".into(), "Let It Be".into()))
        );
        assert_eq!(
            p("03 - The Beatles - Let It Be"),
            Some(("The Beatles".into(), "Let It Be".into()))
        );
        assert_eq!(
            p("03. Nirvana - Polly"),
            Some(("Nirvana".into(), "Polly".into()))
        );
        assert_eq!(
            p("1-07 AC/DC - High Voltage"),
            Some(("AC/DC".into(), "High Voltage".into()))
        );
        assert_eq!(p("FMP_C5_Audio_Beatles_LetItBe"), None);
        assert_eq!(p("Track 01"), None);
    }

    #[test]
    fn years() {
        assert_eq!(parse_year("1970"), Some(1970));
        assert_eq!(parse_year("1970-05-08"), Some(1970));
        assert_eq!(parse_year("08/05/1970"), Some(1970));
        assert_eq!(parse_year("12345"), None);
        assert_eq!(parse_year("n/a"), None);
    }

    #[test]
    fn missing_file_gives_empty_tags() {
        assert_eq!(
            read_tags(Path::new("/definitely/not/here.mp3")),
            Tags::default()
        );
    }
}
