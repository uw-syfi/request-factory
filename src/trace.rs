use anyhow::{anyhow, Context, Result};
use serde::Deserialize;
use std::collections::BTreeMap;

/// One replayable round from the session trace CSV.
#[derive(Debug, Clone, Deserialize)]
pub(crate) struct SessionStep {
    #[serde(alias = "id")]
    pub(crate) session_id: String,
    #[serde(default)]
    pub(crate) arrival_time: f64,
    pub(crate) round_idx: usize,
    pub(crate) prefix_len: usize,
    pub(crate) input_len: usize,
    pub(crate) output_len: usize,
    pub(crate) tool_wait_after_ms: f64,
}

/// The result of rescaling trace arrival offsets to a requested session rate.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ArrivalRateAdjustment {
    pub(crate) trace_rate: f64,
    pub(crate) target_rate: f64,
    pub(crate) time_scale: f64,
}

/// Load the trace CSV into per-session, round-ordered step lists.
pub(crate) fn load_sessions(
    path: &str,
    max_sessions: Option<usize>,
) -> Result<BTreeMap<String, Vec<SessionStep>>> {
    let mut reader = csv::Reader::from_path(path)
        .with_context(|| format!("failed to open session trace: {path}"))?;
    let mut sessions: BTreeMap<String, Vec<SessionStep>> = BTreeMap::new();

    for row in reader.deserialize() {
        let step: SessionStep = row.context("failed to parse session trace row")?;
        sessions
            .entry(step.session_id.clone())
            .or_default()
            .push(step);
    }

    for steps in sessions.values_mut() {
        steps.sort_by_key(|step| step.round_idx);
    }

    if let Some(max) = max_sessions {
        let keys: Vec<String> = sessions.keys().skip(max).cloned().collect();
        for key in keys {
            sessions.remove(&key);
        }
    }

    Ok(sessions)
}

/// Estimate the trace's session arrival rate from its mean inter-session interval.
///
/// With `n` session starts spanning `t` seconds, the mean interval is `t / (n - 1)`, so the
/// corresponding rate is `(n - 1) / t`. Simultaneous starts contribute zero-length intervals and
/// therefore retain bursts in the estimate and in subsequent time scaling.
pub(crate) fn session_arrival_rate(sessions: &BTreeMap<String, Vec<SessionStep>>) -> Option<f64> {
    let mut arrivals = sessions
        .values()
        .filter_map(|steps| steps.first())
        .map(|step| step.arrival_time.max(0.0));
    let first = arrivals.next()?;
    let mut min_arrival_ms = first;
    let mut max_arrival_ms = first;
    let mut count = 1usize;

    for arrival_ms in arrivals {
        if !arrival_ms.is_finite() {
            return None;
        }
        min_arrival_ms = min_arrival_ms.min(arrival_ms);
        max_arrival_ms = max_arrival_ms.max(arrival_ms);
        count += 1;
    }

    let span_seconds = (max_arrival_ms - min_arrival_ms) / 1000.0;
    if count < 2 || !first.is_finite() || span_seconds <= 0.0 {
        return None;
    }

    Some((count - 1) as f64 / span_seconds)
}

/// Scale every trace arrival offset so its measured session rate equals `target_rate`.
///
/// Multiplying time by `trace_rate / target_rate` is equivalent to dividing each arrival offset
/// by the requested speedup. This retains the trace's relative gaps instead of replacing them with
/// a constant or newly sampled arrival process.
pub(crate) fn apply_session_arrival_rate(
    sessions: &mut BTreeMap<String, Vec<SessionStep>>,
    target_rate: f64,
) -> Result<ArrivalRateAdjustment> {
    if !target_rate.is_finite() || target_rate <= 0.0 {
        return Err(anyhow!(
            "--rate must be a finite value greater than 0, got {target_rate}"
        ));
    }

    let trace_rate = session_arrival_rate(sessions).ok_or_else(|| {
        anyhow!(
            "cannot apply --rate: the selected trace needs at least two sessions with distinct arrival times"
        )
    })?;
    let time_scale = trace_rate / target_rate;

    for steps in sessions.values_mut() {
        for step in steps {
            step.arrival_time *= time_scale;
            if !step.arrival_time.is_finite() {
                return Err(anyhow!(
                    "--rate produced a non-finite arrival time; choose a less extreme value"
                ));
            }
        }
    }

    Ok(ArrivalRateAdjustment {
        trace_rate,
        target_rate,
        time_scale,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn step(session_id: &str, arrival_time: f64, round_idx: usize) -> SessionStep {
        SessionStep {
            session_id: session_id.to_string(),
            arrival_time,
            round_idx,
            prefix_len: 0,
            input_len: 1,
            output_len: 1,
            tool_wait_after_ms: 0.0,
        }
    }

    fn sessions(arrivals: &[f64]) -> BTreeMap<String, Vec<SessionStep>> {
        arrivals
            .iter()
            .enumerate()
            .map(|(index, &arrival_time)| {
                let id = index.to_string();
                (
                    id.clone(),
                    vec![step(&id, arrival_time, 0), step(&id, arrival_time, 1)],
                )
            })
            .collect()
    }

    #[test]
    fn measures_rate_from_mean_inter_session_interval() {
        let sessions = sessions(&[0.0, 500.0, 1_000.0]);
        assert_eq!(session_arrival_rate(&sessions), Some(2.0));
    }

    #[test]
    fn scales_all_rounds_to_the_target_rate() {
        let mut sessions = sessions(&[0.0, 500.0, 1_000.0]);
        let adjustment = apply_session_arrival_rate(&mut sessions, 4.0).unwrap();

        assert_eq!(adjustment.trace_rate, 2.0);
        assert_eq!(adjustment.target_rate, 4.0);
        assert_eq!(adjustment.time_scale, 0.5);
        assert_eq!(session_arrival_rate(&sessions), Some(4.0));
        assert!(sessions["1"].iter().all(|step| step.arrival_time == 250.0));
        assert!(sessions["2"].iter().all(|step| step.arrival_time == 500.0));
    }

    #[test]
    fn preserves_bursts_and_relative_gaps() {
        let mut sessions = sessions(&[100.0, 100.0, 300.0, 900.0]);
        let adjustment = apply_session_arrival_rate(&mut sessions, 10.0).unwrap();

        assert!((adjustment.trace_rate - 3.75).abs() < f64::EPSILON);
        let scaled: Vec<f64> = sessions
            .values()
            .map(|steps| steps[0].arrival_time)
            .collect();
        assert_eq!(scaled, vec![37.5, 37.5, 112.5, 337.5]);
    }

    #[test]
    fn rejects_invalid_target_or_unmeasurable_trace_rate() {
        let mut valid = sessions(&[0.0, 1_000.0]);
        assert!(apply_session_arrival_rate(&mut valid, 0.0).is_err());
        assert!(apply_session_arrival_rate(&mut valid, f64::NAN).is_err());

        let mut simultaneous = sessions(&[0.0, 0.0]);
        assert!(apply_session_arrival_rate(&mut simultaneous, 1.0).is_err());

        let mut single = sessions(&[0.0]);
        assert!(apply_session_arrival_rate(&mut single, 1.0).is_err());
    }
}
