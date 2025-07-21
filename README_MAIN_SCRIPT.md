# F1 Telemetry Data Extraction Pipeline - Main Script

This main script provides a comprehensive command-line interface for extracting F1 telemetry data using the Pipeline module. The script supports multiple extraction modes and saves all data to CSV files for further analysis.

## Features

- **Multiple Extraction Modes**: Single race, season-wide, corner telemetry, and raw telemetry extraction
- **Flexible Driver Selection**: Choose specific drivers or extract data for all drivers
- **Race Selection**: Process individual races or multiple races in a season
- **Automatic Data Alignment**: Uses spline-based distance normalization for consistent driver comparisons
- **CSV Output**: All data is saved in CSV format with timestamps for easy organization

## Installation

Ensure you have the required dependencies installed:

```bash
# If using Poetry (recommended)
poetry install

# Or using pip
pip install fastf1 pandas numpy scipy scikit-learn tqdm matplotlib seaborn plotly openpyxl
```

## Usage

### Basic Commands

```bash
# Make the script executable
chmod +x main.py

# Run the script
python main.py --help
```

### Extraction Modes

#### 1. Single Race Data Extraction
Extract comprehensive track analysis for specific drivers in a single race:

```bash
python main.py --mode single --year 2024 --race "Monaco" --drivers VER,HAM,LEC
```

#### 2. Season Data Extraction
Extract season-wide comprehensive driver features:

```bash
# All drivers, all races
python main.py --mode season --year 2024

# Specific drivers and races
python main.py --mode season --year 2024 --drivers VER,HAM --races "Monaco,Silverstone" --max-races 5
```

#### 3. Corner Telemetry Extraction
Extract corner-specific telemetry data for clustering analysis:

```bash
python main.py --mode corner --year 2024 --drivers VER,HAM,LEC
```

#### 4. Raw Telemetry Extraction
Extract raw telemetry data with spline alignment:

```bash
python main.py --mode raw --year 2024 --race "Monza" --drivers VER,HAM
```

### Utility Commands

#### List Available Races
```bash
python main.py --list-races --year 2024
```

#### List Available Drivers
```bash
python main.py --list-drivers --year 2024 --race "Monaco"
```

### Command Line Options

| Option | Description | Example |
|--------|-------------|---------|
| `--mode` | Extraction mode (single/season/corner/raw) | `--mode season` |
| `--year` | Season year (required) | `--year 2024` |
| `--race` | Single race name | `--race "Monaco"` |
| `--races` | Comma-separated race names | `--races "Monaco,Silverstone"` |
| `--drivers` | Comma-separated driver codes | `--drivers VER,HAM,LEC` |
| `--max-races` | Maximum races to process | `--max-races 10` |
| `--output-dir` | Output directory for CSV files | `--output-dir ./results` |
| `--list-races` | List available races for year | `--list-races --year 2024` |
| `--list-drivers` | List available drivers for race | `--list-drivers --year 2024 --race "Monaco"` |

## Output Files

The script generates timestamped CSV files in the specified output directory:

- **Single Race**: `single_race_{year}_{race}_{timestamp}.csv`
- **Season Features**: `season_features_{year}_{timestamp}.csv`
- **Corner Telemetry**: `corner_telemetry_{year}_{timestamp}.csv`
- **Raw Telemetry**: `raw_telemetry_{year}_{race}_{timestamp}.csv`

## Data Types

### Single Race Data
- Comprehensive track analysis per driver
- Corner performance metrics
- Lap-by-lap analysis features

### Season Features
- Multi-race aggregated driver characteristics
- Statistical performance indicators across the season
- Suitable for clustering analysis

### Corner Telemetry
- Corner-specific driving behavior
- Speed, acceleration, and braking patterns
- Optimized for comparative analysis

### Raw Telemetry
- High-resolution telemetry data with spline alignment
- Complete dataset for detailed analysis
- Includes position data, speeds, and derived metrics

## Technical Details

The pipeline uses **spline-based distance alignment** to ensure consistent driver comparisons:

1. **Reference Spline**: Uses the session's fastest lap to create a reference racing line
2. **Spatial Indexing**: Employs k-d trees for efficient position mapping
3. **Data Alignment**: Maps all telemetry data to common reference points
4. **Statistical Validity**: Enables meaningful cross-driver comparisons

## Driver Codes

Common F1 driver codes (use these with `--drivers`):
- VER (Verstappen), HAM (Hamilton), LEC (Leclerc)
- RUS (Russell), NOR (Norris), PIA (Piastri)
- SAI (Sainz), PER (Perez), ALO (Alonso)
- STR (Stroll), RIC (Ricciardo), TSU (Tsunoda)
- GAS (Gasly), OCO (Ocon), ALB (Albon)
- SAR (Sargeant), MAG (Magnussen), HUL (Hulkenberg)
- BOT (Bottas), ZHO (Zhou), COL (Colapinto)

## Example Workflows

### Complete Season Analysis
```bash
# 1. List available races
python main.py --list-races --year 2024

# 2. Extract season features for top drivers
python main.py --mode season --year 2024 --drivers VER,HAM,LEC,NOR,PIA --max-races 15

# 3. Extract corner telemetry for clustering
python main.py --mode corner --year 2024 --drivers VER,HAM,LEC,NOR,PIA
```

### Single Race Deep Dive
```bash
# 1. Check available drivers
python main.py --list-drivers --year 2024 --race "Monaco"

# 2. Extract comprehensive race data
python main.py --mode single --year 2024 --race "Monaco" --drivers VER,HAM,LEC

# 3. Extract raw telemetry for detailed analysis
python main.py --mode raw --year 2024 --race "Monaco" --drivers VER,HAM
```

## Cache Management

The script automatically creates and manages a FastF1 cache directory (`./cache`) to speed up subsequent data extractions. The cache can grow large over time, so consider periodic cleanup.

## Error Handling

The script includes comprehensive error handling and will:
- Continue processing other drivers if one fails
- Provide detailed progress information
- Log errors while attempting to complete the extraction
- Generate partial results when possible

## Performance Notes

- **Season extractions** can take 30-60 minutes depending on the number of races and drivers
- **Corner telemetry** extractions use parallel processing for better performance
- **Raw telemetry** extractions generate large datasets (100MB+ per race)
- Use `--max-races` to limit processing time for testing

---

For technical details about the underlying algorithms and statistical implications, see `Pipeline/EXTRACTION_TECHNICAL_GUIDE.md`. 