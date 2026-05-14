# Generalized Selective Classification for Medical Imaging

This repository contains the code and notebooks for our course project on generalized selective classification for medical imaging. We evaluate whether medical imaging classifiers can produce useful "I don't know" signals under clean data, label shift, and covariate shift.

## Project Summary

Clinical deep learning models can make confident predictions on shifted or unfamiliar inputs. This is risky in medical imaging because errors have unequal clinical consequences. We adapt the generalized selective classification framework of Liang et al. to medical imaging and evaluate multiple confidence scores across three datasets:

- PathMNIST
- ISIC 2019
- CheXpert Small

We evaluate softmax-based, logit-based, and margin-based confidence scores, including MSP, calibrated MSP, entropy, Energy, ODIN, Doctor, RLconf, and RLgeo.

## Main Findings

- No single confidence score performs best across all datasets.
- Energy performs well on PathMNIST.
- RLgeo-3 and calibrated MSP perform best on ISIC 2019.
- MSP performs best on CheXpert Small.
- Severity-weighted AURC shows that uniform metrics can underestimate clinical risk.

## Repository Structure

```text
notebooks/
  PathMNIST.ipynb
  ISIC2019.ipynb
  CheXpert_Small.ipynb

results/
  Results tables and figures used in the report.

docs/
  Final report PDF.