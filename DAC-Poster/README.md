# CREDIT DAC Poster

The poster is generated from the official `DACYF-PosterFormat.pptx` template and preserves its 36 x 48 inch portrait layout.

## Output

- `output/CREDIT_DAC_Young_Fellows_Poster.pptx`: editable PowerPoint poster
- `output/CREDIT_DAC_Young_Fellows_Poster.pdf`: print preview generated with LibreOffice
- `output/CREDIT_DAC_Young_Fellows_Poster.png`: raster preview

## Regenerate

The script reads the current RTX 5090 and H100 evaluation CSV files, regenerates the poster-specific plots, and then updates a copy of the supplied template.

```bash
PYTHONPATH=/tmp/dac-poster-pkgs conda run -n cluster \
  python DAC-Poster/create_poster.py
```

The PowerPoint contains `Fellow ID: ______` because the ID was not present in the paper or workspace. Replace that field before printing.
