# feature_windows.parquet — optional deposit

The manuscript states that this array is regenerated rather than archived. If
you upload it anyway, amend the Data Availability Statement accordingly, because
it currently says the opposite.

## Schema

| column | dtype | description |
|---|---|---|
| `operator_id` | int8 | 1–24 |
| `session_id` | int8 | 1–8 |
| `window_idx` | int16 | 0–1535 within the session |
| `t_start_s` | float32 | seconds from shift start |
| `arm` | category | `baseline` or `intervention` |
| `fold` | int8 | 0–5, the fold in which this operator was held out |
| `f000` … `f126` | float32 | the 127 features, in the order of Table 2 |
| `fdi` | float32 | Fatigue Demand Index, 0–100 |
| `label` | category | `fresh`, `moderate`, `severe` |

Rows: 294,912. Expected size roughly 150 MB in Parquet with Snappy compression,
against about 1.2 GB as CSV.

## Writing it

```python
df.to_parquet("feature_windows.parquet", compression="snappy", index=False)
```

Keep `float32`. The features carry no more than six significant digits, and
`float64` would double the file for no gain.

## Column order

`f000`–`f126` must follow Table 2: 40 sEMG time-domain, 24 sEMG frequency-domain,
36 IMU kinematic, 12 biomechanical, 12 FSR, 3 contextual. Ship
`feature_names.csv` mapping each index to its name, channel and unit; without it
the array is close to unusable by anyone else.
