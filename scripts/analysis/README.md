# Analysis Scripts

Post-processing and analysis scripts for CWSN results.

## Available Analysis Scripts

| Script | Purpose |
|--------|---------|
| **master_analysis.py** | Comprehensive analysis of all chains and results |
| **analyze_chain_detailed.py** | Detailed chain diagnostics |
| **improved_analysis.py** | Enhanced analysis with additional metrics |
| **analyze_usage.py** | Memory and resource usage analysis |
| **check_logp.py** | Validate log-probability calculations |
| **make_cov.py** | Generate covariance matrices from chains |

## Usage

Run individual scripts from the repository root:

```bash
# Master analysis (recommended)
python scripts/analysis/master_analysis.py

# Detailed chain analysis
python scripts/analysis/analyze_chain_detailed.py

# Other analyses
python scripts/analysis/<script_name>.py
```

## Output Locations

Analysis outputs are saved to:
- `results/` - Summary statistics and reports
- `data/desi/edr/processed/mcmc_analysis/` - Chain diagnostics
- Log files in current directory

## Notes

- These scripts operate on chain files and should be run **after** MCMC inference is complete
- Some scripts require specific data files (see main README.md for details)
- Analysis outputs will be automatically organized in `results/` directory
