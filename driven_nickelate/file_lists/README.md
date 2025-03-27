# Data descriptions

All data is acquired by electro-optic sampling of the _probe_---a THz pulse. This sampling uses a _gate pulse_ which travels collinearly with the THz probe in an electro-optic crystal, and undergoes a change of polarization (acquires ellipticity) due to the electric field of the THz pulse. Stepping a movable mirror, the _delay stage_, selects the time slot on the THz probe waveform that is being sampled. The ellipticity is detected using a combination of a quarter-wave plate and a Wollaston polarizing prism, splitting the light into two polarization component whose relative intensity is detected using a pair of balanced photodiodes. The resulting voltage difference is sent through a gated integrator, followed by a lock-in detector. The latter is used to demodulate the signal, which was modulated either by optical chopping (blocking every N pulse, for example), or merely by the reptition rate of the laser (1 kHz).

## Samples

All films were grown using pulsed laser deposition.

| Name      | Description                                                                                |
| --------- | ------------------------------------------------------------------------------------------ |
| F21059    | Au eSRR metasurface / SrTiO<sub>3</sub> (5 uc) / NdNiO<sub>3</sub> (30 uc) / LSAT (0.5 mm) |
| F21062    | SrTiO<sub>3</sub> (5 uc) / NdNiO<sub>3</sub> (30 uc) / LSAT (0.5 mm)                       |
| LSAT_eSRR | Au eSRR metasurface / LSAT (0.5 mm)                                                        |
| LSAT      | LSAT (0.5 mm)                                                                              |

## Terahertz crystals

All crystals have a 6 mm aperture diameter.

| Shortname | Description                         | Note                  |
| --------- | ----------------------------------- | --------------------- |
| ZT-0.5    | 0.5 mm ZnTe (110)                   |                       |
| ZT-1.0    | 1.0 mm ZnTe (110)                   |                       |
| GP        | 0.2 mm GaP (110) / 1.0 mm GaP (100) | Optically contacted   |
| PNPA      | ~1.0 mm PNPA                        | Terahertz Innovations |

## File lists

We use the following abbreviations in the tables below:

| Abbrev. | Description                                      | Note                                   |
| ------- | ------------------------------------------------ | -------------------------------------- |
| P       | Polarizer (Glan-Taylor)                          | Before PNPA crystal                    |
| HWP     | Half-wave plate (zero-order)                     | Before P. Controls generator power.    |
| PP      | Delay of THz probe and gate relative to THz pump | Long stage, before small sampler stage |
| GP      | Delay of gate relative to THz probe              | Small stage                            |
| T       | Temperature (Montana cryostat)                   |                                        |

### linear_probe

A 500 Hz modulation of the THz probe (optical chopper) is used for lock-in detection. The varied parameter for the linear spectroscopy measurements is only the temperature.

| Filename          | Sample    | Detector | Source |
| ----------------- | --------- | -------- | ------ |
| xT_LSAT_eSRR      | LSAT_eSRR | GP       | ZT-1.0 |
| xT_LSAT           | LSAT      | GP       | ZT-1.0 |
| xT_NNO_eSRR       | F21059    | GP       | ZT-1.0 |
| xT_NNO            | F21062    | GP       | ZT-1.0 |
| xT_weakfield_PNPA | F21059    | ZnTe-0.5 | PNPA   |
| xT_weakfield_ZnTe | F21059    | ZnTe-0.5 | ZT-1.0 |
| xT_ZnTe_LSAT      | LSAT      | GP       | ZT-1.0 |
| xT_ZnTe_NNO_eSRR  | F21059    | GP       | ZT-1.0 |
| xT_ZnTe_NNO       | F21062    | GP       | ZT-1.0 |

### nonlinear_probe

| Filename   | Sample | Parameter | Detector | Source | Pump rep. | Probe rep. | Demod. |
| ---------- | ------ | --------- | -------- | ------ | --------- | ---------- | ------ |
| xHWPxT_REF | None   | HWP       | GP       | PNPA   | N/A       | 500 Hz     | 500 Hz |
| xHWPxT     | F21059 | HWP, T    | GP       | PNPA   | N/A       | 500 Hz     | 500 Hz |

### nir_pump

The sample F21052 is used for these measurements.

| Filename                              | Parameter | Detector | Probe  | Pump      | Note                  |
| ------------------------------------- | --------- | -------- | ------ | --------- | --------------------- |
| xPP_ProbeScan_PumpNIR                 | PP        | GP       | ZT-1.0 | OPA 1.3um |                       |
| xPP_ProbeScan_PumpNIRHoriz            | PP        | GP       | ZT-1.0 | OPA 1.3um | 90° rotation of probe |
| 050323_xPP_TransientProbeTraces       | PP        | GP       | ZT-1.0 | OPA 1.3um |                       |
| 040323_xPP_TransientProbeScan_PumpNIR | PP        | GP       | ZT-1.0 | OPA 1.3um |                       |

### pump_probe

The terahertz probe (weak) pulses were generated with ZT-1.0, and the THz pump pulses with PNPA.

| Filename                         | Variable | Parameters | Detector |
| -------------------------------- | -------- | ---------- | -------- |
| 010323_xTxPP1DScan               |          | T, PP      | GP       |
| 020323_xHWPxT_fastoscillations   |          | HWP, T     | GP       |
| 030223_xPP_ChoppedProbe          |          | PP         | GP       |
| 050223_xPP_xHWP                  |          | PP         | GP       |
| 070223_xPP_xHWP_xT               |          | PP, HWP, T | GP       |
| 080323_xHWPxTProbeScan10ps       |          | HWP, T     | GP       |
| 090323_xHWPxPPProbeScanHorizPump |          | HWP, PP    | GP       |
| 090323_xPP_ProbeScan_LPF         |          | PP         | GP       |
| 100323_xPP_ProbeScan_noLPF       |          | PP         | GP       |
| 110323_xGenPower_earlytimes      |          | HWP        | ZT-0.5   |
| 120323_xPP_LSAT_eSRR             |          | PP         | ZT-0.5   |
| 180223_xPP_GateTrace             |          | PP         | GP       |
| 190323_reflection_xHWP_GateScan  |          | HWP        | ZT-0.5   |
| 200223_hystereses                |          | HWP        | GP       |
| 210223_xT_ProbeScan              |          | T          | GP       |
| 220123_xPP_ChoppedProbe          |          | PP         | GP       |
| 220223_2DProbeScan               |          | PP         | GP       |
| 220223_xT_LSAT                   |          | T          | GP       |
| 230223_xHWP_100ps_ProbeScan      |          | HWP        | GP       |
| 240223_xHWPxT                    |          | T, HWP     | GP       |
| 260223_xHWPfine                  |          | HWP        | GP       |
| 270223_xHWP1DTrace10ps           |          | HWP        | GP       |
| reflection_xPP                   |          | PP         | ZT-0.5   |
| reflection_xT                    |          | T          | ZT-0.5   |
