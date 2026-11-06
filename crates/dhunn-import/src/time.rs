//! Minimal UTC timestamp parsing/formatting for export files (no timezone database needed:
//! exports use UTC or explicit offsets).

/// Days since 1970-01-01 for a proleptic Gregorian date (Howard Hinnant's algorithm).
pub fn days_from_civil(y: i64, m: u32, d: u32) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let m = m as i64;
    let doy = (153 * (if m > 2 { m - 3 } else { m + 9 }) + 2) / 5 + d as i64 - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

/// `(year, month, day)` for days since 1970-01-01.
pub fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (yoe + era * 400 + i64::from(m <= 2), m, d)
}

fn num(s: &str) -> Option<i64> {
    (!s.is_empty() && s.bytes().all(|b| b.is_ascii_digit()))
        .then(|| s.parse().ok())
        .flatten()
}

/// Parses `YYYY-MM-DD[(T| )HH:MM[:SS[.fff]]][Z|±HH:MM|±HHMM]` or `YYYYMMDD` into Unix seconds.
pub fn parse_timestamp(s: &str) -> Option<i64> {
    let s = s.trim();
    let (date, rest) = if s.len() == 8 && num(s).is_some() {
        (format!("{}-{}-{}", &s[..4], &s[4..6], &s[6..]), "")
    } else {
        let cut = s.find(['T', ' ']).unwrap_or(s.len());
        (s[..cut].to_owned(), s.get(cut + 1..).unwrap_or(""))
    };
    let mut parts = date.split('-');
    let (y, m, d) = (
        num(parts.next()?)?,
        num(parts.next()?)?,
        num(parts.next()?)?,
    );
    if parts.next().is_some() || !(1..=12).contains(&m) || !(1..=31).contains(&d) {
        return None;
    }
    let mut secs = days_from_civil(y, m as u32, d as u32) * 86_400;

    if rest.is_empty() {
        return Some(secs);
    }
    // Split off a timezone suffix.
    let (clock, offset) = if let Some(c) = rest.strip_suffix('Z') {
        (c, 0)
    } else if let Some(i) = rest.rfind(['+', '-']) {
        let tz = rest[i + 1..].replace(':', "");
        if tz.len() != 4 {
            return None;
        }
        let sign = if rest.as_bytes()[i] == b'-' { -1 } else { 1 };
        (
            &rest[..i],
            sign * (num(&tz[..2])? * 3600 + num(&tz[2..])? * 60),
        )
    } else {
        (rest, 0)
    };
    let clock = clock.split('.').next()?;
    let mut hms = clock.split(':');
    let h = num(hms.next()?)?;
    let mi = num(hms.next()?)?;
    let se = hms.next().map(num).unwrap_or(Some(0))?;
    if h > 23 || mi > 59 || se > 60 {
        return None;
    }
    secs += h * 3600 + mi * 60 + se - offset;
    Some(secs)
}

/// `YYYY-MM-DDTHH:MM:SSZ` (Spotify's format).
pub fn format_timestamp(ts: i64) -> String {
    let (y, m, d) = civil_from_days(ts.div_euclid(86_400));
    let s = ts.rem_euclid(86_400);
    format!(
        "{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}Z",
        s / 3600,
        s % 3600 / 60,
        s % 60
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_instants() {
        assert_eq!(parse_timestamp("1970-01-01T00:00:00Z"), Some(0));
        assert_eq!(parse_timestamp("2026-03-01T18:04:12Z"), Some(1_772_388_252));
        assert_eq!(parse_timestamp("2026-03-01 18:04"), Some(1_772_388_240));
        assert_eq!(
            parse_timestamp("2026-03-01T18:04:12.999Z"),
            Some(1_772_388_252)
        );
        assert_eq!(
            parse_timestamp("2026-03-01T23:34:12+05:30"),
            Some(1_772_388_252)
        );
        assert_eq!(
            parse_timestamp("2026-03-01T13:04:12-0500"),
            Some(1_772_388_252)
        );
        assert_eq!(parse_timestamp("20260301"), Some(1_772_323_200));
        assert_eq!(parse_timestamp("2026-13-01"), None);
        assert_eq!(parse_timestamp("yesterday"), None);
    }

    #[test]
    fn roundtrip() {
        for ts in [0, 951_782_400, 1_772_388_252, 4_102_444_799] {
            assert_eq!(parse_timestamp(&format_timestamp(ts)), Some(ts));
        }
        assert_eq!(format_timestamp(1_772_388_252), "2026-03-01T18:04:12Z");
    }
}
