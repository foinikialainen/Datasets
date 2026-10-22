# GNSSLogger_data.zip Dataset
This repository contains raw Global Navigation Satellite System (GNSS) measurement datasets collected using the Android **GNSS Logger** application in two different transportation scenarios:
- **Car:**
    - Location: Drive in Espoo.
- **Train:**
    - Location: Train ride in Frace. The train passes through tunnels, so GNSS is not available for these instances    

## Dataset Structure
```text
GNSSLogger_data/
├── Car/
│   ├── gnss_log_2026_08_23_18_00_08.txt
│   ├── gnss_log_2026_08_23_18_00_08.26o
│   ├── gnss_log_2026_08_23_18_00_08.nmea
│   ├── BRDM00DLR_S_20262350000_01D_MN.rnx
│   └── IGS0OPSFIN_20262350000_01D_15M_ORB.SP3
│
└── Train/
    ├── gnss_log_2026_09_19_09_05_09.txt
    ├── gnss_log_2026_09_19_09_05_09.26o
    ├── gnss_log_2026_09_19_09_05_09.nmea
    ├── BRDM00DLR_S_20262620000_01D_MN.rnx
    └── IGS0OPSFIN_20262620000_01D_15M_ORB.SP3
```

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

