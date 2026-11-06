# Data and model sources

No dataset text and no model weight is redistributed in this repository. The manifests in `data/manifests/` contain identifiers (query / product / passage ids, labels, row order, checksums) and short derived fields only; texts are restored from the sources below.
A few small files derived from the public datasets are kept because the analysis needs them: the Base model's validation scores (`cpmr_experiment/*/p1_T*/p1_raw_cand*.json.gz`, keyed by ESCI query and product id) and the APO-CF proposer exchange (`p1_exchange.json`, which quotes the instruction texts and the diagnostic confusion matrix, no dataset rows).

| resource | source | terms |
|---|---|---|
| ESCI / Shopping Queries Dataset (US, small version) | https://github.com/amazon-science/esci-data ; Reddy et al., 2022, arXiv:2206.06588 | Apache License 2.0 (check the repository for the current terms) |
| MS MARCO passage ranking (collection, train queries) | https://microsoft.github.io/msmarco/ ; Bajaj et al., 2016 | non-commercial research use under the terms published by the MS MARCO owners; obtained through `ir_datasets` (`msmarco-passage`) |
| TREC Deep Learning 2019 / 2020 passage judgments | https://microsoft.github.io/msmarco/TREC-Deep-Learning ; Craswell et al., 2020 / 2021 | as published by NIST / the track organizers; obtained through `ir_datasets` (`msmarco-passage/trec-dl-2019/judged`, `msmarco-passage/trec-dl-2020/judged`) |
| Qwen3-4B | https://huggingface.co/Qwen/Qwen3-4B | Apache License 2.0 |
| Llama-3.1-8B-Instruct | https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct | Llama 3.1 Community License (gated; accept on Hugging Face) |
| Gemma-3-4B-it | https://huggingface.co/google/gemma-3-4b-it | Gemma Terms of Use (gated; accept on Hugging Face) |

The APO-CF meta-prompt in `src/apocf_prompt.py` follows Choi (2025), *Applied Sciences* 15(9):5198, https://doi.org/10.3390/app15095198 (prompt strings of the original implementation, copied unchanged).
The code license is not yet decided; see the end of `README.md`.
