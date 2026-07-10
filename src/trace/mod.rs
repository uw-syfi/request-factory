mod session;
mod vibesim;

use anyhow::{anyhow, Result};
use clap::ValueEnum;
use std::collections::BTreeMap;

pub(crate) use session::SessionStep;
pub(crate) use vibesim::VibeSimRequest;

/// Source schema selector. Each frontend retains its own typed workload rather
/// than filling absent fields in a universal row with zeros or nulls.
#[derive(ValueEnum, Clone, Copy, Debug)]
pub(crate) enum TraceFormat {
    Session,
    Vibesim,
}

/// Typed replay plans produced by the trace frontends.
pub(crate) enum ReplayWorkload {
    Sessions(BTreeMap<String, Vec<SessionStep>>),
    IndependentRequests(Vec<VibeSimRequest>),
}

/// The result of rescaling trace arrival offsets to a requested session rate.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ArrivalRateAdjustment {
    pub(crate) trace_rate: f64,
    pub(crate) target_rate: f64,
    pub(crate) time_scale: f64,
}

/// Dispatch to one source-specific frontend without erasing its request type.
pub(crate) fn load_workload(
    path: &str,
    format: TraceFormat,
    max_items: Option<usize>,
) -> Result<ReplayWorkload> {
    match format {
        TraceFormat::Session => session::load(path, max_items).map(ReplayWorkload::Sessions),
        TraceFormat::Vibesim => {
            vibesim::load(path, max_items).map(ReplayWorkload::IndependentRequests)
        }
    }
}

impl ReplayWorkload {
    pub(crate) fn unit_count(&self) -> usize {
        match self {
            Self::Sessions(sessions) => sessions.len(),
            Self::IndependentRequests(requests) => requests.len(),
        }
    }

    pub(crate) fn unit_label(&self) -> &'static str {
        match self {
            Self::Sessions(_) => "sessions",
            Self::IndependentRequests(_) => "requests",
        }
    }

    pub(crate) fn arrival_rate(&self) -> Option<f64> {
        match self {
            Self::Sessions(sessions) => arrival_rate(
                sessions
                    .values()
                    .filter_map(|steps| steps.first())
                    .map(|step| step.arrival_time),
            ),
            Self::IndependentRequests(requests) => {
                arrival_rate(requests.iter().map(|request| request.arrival_time))
            }
        }
    }

    pub(crate) fn apply_arrival_rate(&mut self, target_rate: f64) -> Result<ArrivalRateAdjustment> {
        let trace_rate = self.arrival_rate().ok_or_else(|| {
            anyhow!(
                "cannot apply --rate: the selected workload needs at least two items with distinct arrival times"
            )
        })?;
        validate_target_rate(target_rate)?;
        let time_scale = trace_rate / target_rate;
        if !time_scale.is_finite() {
            return Err(anyhow!(
                "--rate produced a non-finite arrival-time scale; choose a less extreme value"
            ));
        }
        match self {
            Self::Sessions(sessions) => {
                for step in sessions.values_mut().flatten() {
                    step.arrival_time *= time_scale;
                }
            }
            Self::IndependentRequests(requests) => {
                for request in requests {
                    request.arrival_time *= time_scale;
                }
            }
        }
        Ok(ArrivalRateAdjustment {
            trace_rate,
            target_rate,
            time_scale,
        })
    }
}

/// Estimate the workload's arrival rate from its mean inter-item interval.
///
/// With `n` starts spanning `t` seconds, the mean interval is `t / (n - 1)`, so the
/// corresponding rate is `(n - 1) / t`. Simultaneous starts contribute zero-length intervals and
/// therefore retain bursts in the estimate and in subsequent time scaling.
fn arrival_rate(arrivals: impl Iterator<Item = f64>) -> Option<f64> {
    let mut arrivals = arrivals.map(|arrival| arrival.max(0.0));
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

fn validate_target_rate(target_rate: f64) -> Result<()> {
    if !target_rate.is_finite() || target_rate <= 0.0 {
        return Err(anyhow!(
            "--rate must be a finite value greater than 0, got {target_rate}"
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

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

    fn sessions(arrivals: &[f64]) -> ReplayWorkload {
        ReplayWorkload::Sessions(
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
                .collect(),
        )
    }

    #[test]
    fn measures_rate_from_mean_inter_session_interval() {
        let sessions = sessions(&[0.0, 500.0, 1_000.0]);
        assert_eq!(sessions.arrival_rate(), Some(2.0));
    }

    #[test]
    fn scales_all_rounds_to_the_target_rate() {
        let mut sessions = sessions(&[0.0, 500.0, 1_000.0]);
        let adjustment = sessions.apply_arrival_rate(4.0).unwrap();

        assert_eq!(adjustment.trace_rate, 2.0);
        assert_eq!(adjustment.target_rate, 4.0);
        assert_eq!(adjustment.time_scale, 0.5);
        assert_eq!(sessions.arrival_rate(), Some(4.0));
        let ReplayWorkload::Sessions(sessions) = sessions else {
            panic!()
        };
        assert!(sessions["1"].iter().all(|step| step.arrival_time == 250.0));
        assert!(sessions["2"].iter().all(|step| step.arrival_time == 500.0));
    }

    #[test]
    fn preserves_bursts_and_relative_gaps() {
        let mut sessions = sessions(&[100.0, 100.0, 300.0, 900.0]);
        let adjustment = sessions.apply_arrival_rate(10.0).unwrap();

        assert!((adjustment.trace_rate - 3.75).abs() < f64::EPSILON);
        let ReplayWorkload::Sessions(sessions) = sessions else {
            panic!()
        };
        let scaled: Vec<f64> = sessions
            .values()
            .map(|steps| steps[0].arrival_time)
            .collect();
        assert_eq!(scaled, vec![37.5, 37.5, 112.5, 337.5]);
    }

    #[test]
    fn rejects_invalid_target_or_unmeasurable_trace_rate() {
        let mut valid = sessions(&[0.0, 1_000.0]);
        assert!(valid.apply_arrival_rate(0.0).is_err());
        assert!(valid.apply_arrival_rate(f64::NAN).is_err());

        let mut simultaneous = sessions(&[0.0, 0.0]);
        assert!(simultaneous.apply_arrival_rate(1.0).is_err());

        let mut single = sessions(&[0.0]);
        assert!(single.apply_arrival_rate(1.0).is_err());
    }

    #[test]
    fn vibesim_frontend_keeps_independent_request_type() {
        let path = std::env::temp_dir().join(format!(
            "tracelab_vibesim_frontend_{}.csv",
            std::process::id()
        ));
        fs::write(
            &path,
            "id,input_len,output_len,arrival_time\nreq-1,16,4,12.5\n",
        )
        .unwrap();
        let workload = load_workload(path.to_str().unwrap(), TraceFormat::Vibesim, None).unwrap();
        fs::remove_file(path).unwrap();

        let ReplayWorkload::IndependentRequests(requests) = workload else {
            panic!("VibeSim frontend must not coerce requests into session rounds")
        };
        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0].id, "req-1");
        assert_eq!(requests[0].input_len, 16);
        assert_eq!(requests[0].output_len, 4);
        assert_eq!(requests[0].arrival_time, 12.5);
    }
}
