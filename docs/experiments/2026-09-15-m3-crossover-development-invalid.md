# M3 crossover development — invalid stopped study

M3 stopped after seven accepted cells when `m3-dev-qps2-a60-seq4` failed the preregistered
lifecycle-cleanliness gate. The run completed, but startup logged Hugging Face metadata-request
timeouts and DNS failures. Its dispatch evidence was valid; performance outcomes were not inspected.

The driver stopped as designed. No retry was allowed because the sole permitted retry condition was
dispatch drift. All raw evidence is preserved under `runs/m3-crossover-development/`; none of these
runs may be used in M3b or later confirmatory scoring.

M3b is a fresh development study with new prompt and arrival seeds. It additionally fixes
`HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` before execution so cached model loading cannot make
network metadata requests. This is an infrastructure correction, not an outcome-driven design
change.
