## Data Content description
The following files are obtained from the GNSSLogger software:

### Raw GNSS Log (`*.txt`)
Original Android GNSS Logger output containing raw GNSS and inertail (accelerometer, gyroscope, magnetometer) measurements and sensor information.

### GNSS Observation File (`*.26o`)
RINEX observation file generated from the GNSS Logger measurements. These files contain pseudorange, carrier-phase, Doppler, and signal strength observations recorded from visible GNSS satellites.

### NMEA Log (`*.nmea`)
Messages produced by the receiver (in standardized NMEA), including position, velocity, time, satellite visibility, and navigation information.

## Supplementary data
This data is not obtained from GNSSLogger but can be useful for GNSS post-processing and positioning analyses.

### Broadcast Ephemeris (`BRDM*.rnx`)
Multi-GNSS broadcast navigation messages in RINEX format. These files provide satellite orbit and clock information transmitted by the satellites and are required for standalone positioning solutions.

### Precise Orbit File (`*.SP3`)
Precise satellite orbit products obtained from the International GNSS Service (IGS). These files enable high-accuracy post-processing techniques such as Precise Point Positioning (PPP).

## Comparing the two apps (GNSSLogger vs. the second app)

`compare_apps.py` compares the data recorded simultaneously by GNSSLogger (`GNSSLogger/`)
and by the second app (`MIMIR/`). You only pass the two parent directories - the script
finds all data files inside them by itself (GNSSLogger `.txt` logs, or the second app's
fixed-size chunk files `log_*.txt`; if the app also wrote a merged `*.csv`, it is parsed
and cross-checked against the chunks):

```bash
pip install -r requirements.txt
python3 compare_apps.py GNSSLogger MIMIR --out comparison_output
```

It focuses on the IMU data (accelerometer / gyroscope / magnetometer): it plots both apps'
signals on the same time axis, compares the values sample by sample, and checks each app's
IMU clock for gaps (periods where nothing was logged for a couple of seconds). It also
compares the GNSS fix data (~1 Hz position estimates): position track, clock jumps to 1970
(timestamps near zero) and other clock jumps, and the position difference between the two
apps. Results are printed to the screen, saved to `comparison_output/report.txt`, and
plotted as PNG files in `comparison_output/`.

Note on time: all comparisons use the Unix-millisecond timestamps stored inside each data
line. File names are only used to sort files: GNSSLogger file names are in UTC, the second
app's file names are in the phone's local time (the script prints the measured difference).

