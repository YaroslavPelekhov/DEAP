"""
EEG channel definitions:
  - DEAP_CHANNELS: 32-channel 10-20 names in DEAP order (index 0-31)
  - BAROMETER_CHANNELS: 20-channel set from the NeuroBarometer device
  - BAROMETER_IN_DEAP: BAROMETER channels that exist in DEAP (with old→new name remapping)
  - DEAP_BAROMETER_INDICES: indices into DEAP's 32 channels for the shared set

Old 10-20 name mapping:
  T3→T7, T4→T8, T5→P7, T6→P8

Barometer channel 'POz' has no exact DEAP counterpart — omitted from intersection.
"""

DEAP_CHANNELS = [
    'Fp1', 'AF3', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7',
    'CP5', 'CP1', 'P3', 'P7', 'PO3', 'O1', 'Oz', 'Pz',
    'P4', 'P8', 'PO4', 'O2', 'T8', 'CP6', 'CP2', 'Cz',
    'C4', 'FC6', 'FC2', 'F8', 'F4', 'AF4', 'Fp2', 'Fz',
]

BAROMETER_CHANNELS_RAW = [
    'Fp1', 'F7', 'F8', 'T4', 'T6', 'T5', 'T3', 'Fp2',
    'O1', 'P3', 'Pz', 'F3', 'Fz', 'F4', 'C4', 'P4',
    'POz', 'C3', 'Cz', 'O2',
]

# Map old names to new 10-20 names
_OLD_TO_NEW = {'T3': 'T7', 'T4': 'T8', 'T5': 'P7', 'T6': 'P8'}
BAROMETER_CHANNELS = [_OLD_TO_NEW.get(ch, ch) for ch in BAROMETER_CHANNELS_RAW]

# Intersection: Barometer channels that exist in DEAP (POz excluded)
BAROMETER_IN_DEAP = [ch for ch in BAROMETER_CHANNELS if ch in DEAP_CHANNELS]

# Indices into DEAP's 32-channel array for the shared channels
DEAP_BAROMETER_INDICES = [DEAP_CHANNELS.index(ch) for ch in BAROMETER_IN_DEAP]

# Frontal channels for FAA/FTA (Frontal Alpha/Theta Asymmetry)
FAA_LEFT  = [DEAP_CHANNELS.index('F3')]
FAA_RIGHT = [DEAP_CHANNELS.index('F4')]
FTA_LEFT  = [DEAP_CHANNELS.index('F3')]
FTA_RIGHT = [DEAP_CHANNELS.index('F4')]

if __name__ == '__main__':
    print(f"DEAP channels ({len(DEAP_CHANNELS)}): {DEAP_CHANNELS}")
    print(f"Barometer raw ({len(BAROMETER_CHANNELS_RAW)}): {BAROMETER_CHANNELS_RAW}")
    print(f"Barometer new names ({len(BAROMETER_CHANNELS)}): {BAROMETER_CHANNELS}")
    print(f"Intersection ({len(BAROMETER_IN_DEAP)}): {BAROMETER_IN_DEAP}")
    print(f"DEAP indices: {DEAP_BAROMETER_INDICES}")
