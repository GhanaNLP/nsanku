# Nsanku

A project by Ghana NLP to test the machine translation performance of Large Language Models (LLMs) on Ghanaian languages.

## Project Overview

Nsanku is an initiative that evaluates how well various open-source language models perform when working with Ghanaian languages. **We have evaluated the performance of over 14 open-source and closed models across 43 languages, using at least 300 sentences per language** Paper publication coming soon.

**Note about data:** Sentences used in the current evaluation were sourced from biblical text.

## Why Nsanku Matters

As AI engineers work to bring Ghanaian languages into large language models, it’s essential to have reliable evidence on how existing models perform. Nsanku provides insights that help developers identify which LLMs are most suitable for use in translating content in Ghanaian languages, and which languages currently have stronger or weaker support.

## Results (Overview)

<div style="display:flex; flex-wrap:wrap; gap:10px;">

<a href="https://github.com/GhanaNLP/nsanku/raw/main/reports_combined/model_performance.png">
  <img src="https://github.com/GhanaNLP/nsanku/raw/main/reports_combined/model_performance.png" width="45%" alt="Overall BLEU Scores">
</a>

<a href="https://github.com/GhanaNLP/nsanku/raw/main/reports_combined/language_performance.png">
  <img src="https://github.com/GhanaNLP/nsanku/raw/main/reports_combined/language_performance.png" width="45%" alt="Overall chrF Scores">
</a>

<a href="https://github.com/GhanaNLP/nsanku/raw/main/reports_combined/model_performance_vs_consistency_quadrant.png">
  <img src="https://github.com/GhanaNLP/nsanku/raw/main/reports_combined/model_performance_vs_consistency_quadrant.png" width="45%" alt="Overall Metrics Comparison">
</a>

<a href="https://github.com/GhanaNLP/nsanku/raw/main/reports_combined/language_performance_vs_consistency_quadrant.png">
  <img src="https://github.com/GhanaNLP/nsanku/raw/main/reports_combined/language_performance_vs_consistency_quadrant.png" width="45%" alt="Overall Average Performance">
</a>

</div>

## Language Specific Results

Detailed performance breakdown for each of the 43 evaluated languages.
**Note:** Unusually high BLUE scores for some models are most likely not an indicator of their generic performance for the language. Further human evaluation is strongly recommended to confirm these results.

<details>

<summary>Click to expand language-specific charts</summary>

<div style="display:grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 15px; margin-top: 20px;">

<!-- Iterating through all 43 language folders identified -->

<div><strong>Abron</strong>
<img src="https://github.com/GhanaNLP/nsanku/raw/main/reports_combined/abr-eng/performance_comparison.png" width="100%"></div>

<div><strong>Gikyode</strong>
<img src="https://github.com/GhanaNLP/nsanku/raw/main/reports_combined/acd-eng/performance_comparison.png" width="100%"></div>

<div><strong>Dangme</strong>
<img src="https://github.com/GhanaNLP/nsanku/raw/main/reports_combined/ada-eng/performance_comparison.png" width="100%"></div>

<div><strong>Siwu</strong>
<img src="https://github.com/GhanaNLP/nsanku/raw/main/reports_combined/akp-eng/performance_comparison.png" width="100%"></div>

<div><strong>Anyin</strong>
<img src="https://github.com/GhanaNLP/nsanku/raw/main/reports_combined/any-eng/performance_comparison.png" width="100%"></div>

<div><strong>Avatime</strong>
<img src="https://github.com/GhanaNLP/nsanku/raw/main/reports_combined/avn-eng/performance_comparison.png" width="100%"></div>

<div><strong>Bisa</strong>
<img src="https://github.com/GhanaNLP/nsanku/raw/main/reports_combined/bib-eng/performance_comparison.png" width="100%"></div>

<div><strong>Bimoba</strong>
<img src="https://github.com/GhanaNLP/nsanku/raw/main/reports_combined/bim-eng/performance_comparison.png" width="100%"></div>

<div><strong>Southern Birifor</strong>
<img src="https://github.com/GhanaNLP/nsanku/raw/main/reports_combined/biv-eng/performance_comparison.png" width="100%"></div>

<div><strong>Tuwuli</strong>
<img src="https://github.com/GhanaNLP/nsanku/raw/main/reports_combined/bov-eng/performance_comparison.png" width="100%"></div>

<div><strong>Ntcham</strong>
<img src="https://github.com/GhanaNLP/nsanku/raw/main/reports_combined/bud-eng/performance_comparison.png" width="100%"></div>

<div><strong>Buli</strong>
<img src="https://github.com/GhanaNLP/nsanku/raw/main/reports_combined/bwu-eng/performance_comparison.png" width="100%"></div>

<div><strong>Anufo</strong>
<img src="https://github.com/GhanaNLP/nsanku/raw/main/reports_combined/cko-eng/performance_comparison.png" width="100%"></div>

<div><strong>Dagbani</strong>
<img src="https://github.com/GhanaNLP/nsanku/raw/main/reports_combined/dag-eng/performance_comparison.png" width="100%"></div>

<div><strong>Southern Dagaare</strong>
<img src="https://github.com/GhanaNLP/nsanku/raw/main/reports_combined/dga-eng/performance_comparison.png" width="100%"></div>

<div><strong>Ewe</strong>
<img src="https://github.com/GhanaNLP/nsanku/raw/main/reports_combined/ewe-eng/performance_comparison.png" width="100%"></div>

<div><strong>Fante</strong>
<img src="https://github.com/GhanaNLP/nsanku/raw/main/reports_combined/fat-eng/performance_comparison.png" width="100%"></div>

<div><strong>Ga</strong>
<img src="https://github.com/GhanaNLP/nsanku/raw/main/reports_combined/gaa-eng/performance_comparison.png" width="100%"></div>

<div><strong>Gonja</strong>
<img src="https://github.com/GhanaNLP/nsanku/raw/main/reports_combined/gjn-eng/performance_comparison.png" width="100%"></div>

<div><strong>Farefare</strong>
<img src="https://github.com/GhanaNLP/nsanku/raw/main/reports_combined/gur-eng/performance_comparison.png" width="100%"></div>

<div><strong>Hanga</strong>
<img src="https://github.com/GhanaNLP/nsanku/raw/main/reports_combined/hag-eng/performance_comparison.png" width="100%"></div>

<div><strong>Konni</strong>
<img src="https://github.com/GhanaNLP/nsanku/raw/main/reports_combined/kma-eng/performance_comparison.png" width="100%"></div>

<div><strong>Kusaal</strong>
<img src="https://github.com/GhanaNLP/nsanku/raw/main/reports_combined/kus-eng/performance_comparison.png" width="100%"></div>

<div><strong>Lelemi</strong>
<img src="https://github.com/GhanaNLP/nsanku/raw/main/reports_combined/lef-eng/performance_comparison.png" width="100%"></div>

<div><strong>Sekpele</strong>
<img src="https://github.com/GhanaNLP/nsanku/raw/main/reports_combined/lip-eng/performance_comparison.png" width="100%"></div>

<div><strong>Mampruli</strong>
<img src="https://github.com/GhanaNLP/nsanku/raw/main/reports_combined/maw-eng/performance_comparison.png" width="100%"></div>

<div><strong>Deg</strong>
<img src="https://github.com/GhanaNLP/nsanku/raw/main/reports_combined/mzw-eng/performance_comparison.png" width="100%"></div>

<div><strong>Nawuri</strong>
<img src="https://github.com/GhanaNLP/nsanku/raw/main/reports_combined/naw-eng/performance_comparison.png" width="100%"></div>

<div><strong>Chumburung</strong>
<img src="https://github.com/GhanaNLP/nsanku/raw/main/reports_combined/ncu-eng/performance_comparison.png" width="100%"></div>

<div><strong>Nkonya</strong>
<img src="https://github.com/GhanaNLP/nsanku/raw/main/reports_combined/nko-eng/performance_comparison.png" width="100%"></div>

<div><strong>Delo</strong>
<img src="https://github.com/GhanaNLP/nsanku/raw/main/reports_combined/ntr-eng/performance_comparison.png" width="100%"></div>

<div><strong>Nyagbo</strong>
<img src="https://github.com/GhanaNLP/nsanku/raw/main/reports_combined/nyb-eng/performance_comparison.png" width="100%"></div>

<div><strong>Nzema</strong>
<img src="https://github.com/GhanaNLP/nsanku/raw/main/reports_combined/nzi-eng/performance_comparison.png" width="100%"></div>

<div><strong>Esahie</strong>
<img src="https://github.com/GhanaNLP/nsanku/raw/main/reports_combined/sfw-eng/performance_comparison.png" width="100%"></div>

<div><strong>Paasaal</strong>
<img src="https://github.com/GhanaNLP/nsanku/raw/main/reports_combined/sig-eng/performance_comparison.png" width="100%"></div>

<div><strong>Tumulung Sisaala</strong>
<img src="https://github.com/GhanaNLP/nsanku/raw/main/reports_combined/sil-eng/performance_comparison.png" width="100%"></div>

<div><strong>Selee</strong>
<img src="https://github.com/GhanaNLP/nsanku/raw/main/reports_combined/snw-eng/performance_comparison.png" width="100%"></div>

<div><strong>Tafi</strong>
<img src="https://github.com/GhanaNLP/nsanku/raw/main/reports_combined/tcd-eng/performance_comparison.png" width="100%"></div>

<div><strong>Tampulma</strong>
<img src="https://github.com/GhanaNLP/nsanku/raw/main/reports_combined/tpm-eng/performance_comparison.png" width="100%"></div>

<div><strong>Twi</strong>
<img src="https://github.com/GhanaNLP/nsanku/raw/main/reports_combined/twi-eng/performance_comparison.png" width="100%"></div>

<div><strong>Vagla</strong>
<img src="https://github.com/GhanaNLP/nsanku/raw/main/reports_combined/vag-eng/performance_comparison.png" width="100%"></div>

<div><strong>Konkomba</strong>
<img src="https://github.com/GhanaNLP/nsanku/raw/main/reports_combined/xon-eng/performance_comparison.png" width="100%"></div>

<div><strong>Kasem</strong>
<img src="https://github.com/GhanaNLP/nsanku/raw/main/reports_combined/xsm-eng/performance_comparison.png" width="100%"></div>

</div>

</details>

## Contributing

We welcome contributions from the community! To contribute:

1. Run the evaluation using our Google Colab notebook
2. Share your results with us
3. We'll include your findings in our collective results

**Get started with the evaluation notebook:** 

## Evaluated Models

We are running evaluations of these models:

- `gpt-4.1`
- `claude-sonnet-4-5`
- `gemini-2.5-flash`
- `deepseek-v3.1`
- `gemma-2-9b-it`
- `gemma-2-27b-it`
- `gpt-oss-120b`
- `kimi-k2-instruct-0905`
- `llama-3.1-405b-instruct`
- `llama-3.3-70b-instruct`
- `llama-4-maverick-17b-128e-instruct`
- `mistral-medium-3-instruct`
- `qwen3-235b-a22b`
- `qwq-32b`
- `seed-oss-36b-instruct`

## Languages Evaluated

The project evaluated **43 Ghanaian languages**:

| **Language**     | **Language** | **Language**     | **Language**     |
| ---------------- | ------------ | ---------------- | ---------------- |
| Abron            | Gikyode      | Dangme           | Siwu             |
| Anyin            | Avatime      | Bisa             | Bimoba           |
| Southern Birifor | Tuwuli       | Ntcham           | Buli             |
| Anufo            | Dagbani      | Southern Dagaare | Ewe              |
| Fante            | Ga           | Gonja            | Farefare         |
| Hanga            | Konni        | Kusaal           | Lelemi           |
| Sekpele          | Mampruli     | Deg              | Nawuri           |
| Chumburung       | Nkonya       | Delo             | Nyagbo           |
| Nzema            | Esahie       | Paasaal          | Tumulung Sisaala |
| Selee            | Tafi         | Tampulma         | Twi              |
| Vagla            | Konkomba     | Kasem            |                  |

## Contributors

Thanks to our awesome contributors who made it possible:

1. [Onesimus Addo Appiah](https://www.linkedin.com/in/onesimus-appiah/)
2. [Mich-Seth Owusu](https://www.linkedin.com/in/mich-seth-owusu/)
3. [Jonathan Asiamah](https://www.linkedin.com/in/jonathan-asiamah-4639a5147/)
4. [Elias Dzobo](https://www.linkedin.com/in/eliasdzobo/)
5. [Kelvin Newman](https://www.linkedin.com/in/kelvin-newman-09b961255/)
6. [Edmund O. Benefo](https://www.linkedin.com/in/edmund-o-benefo/)
7. [Gerhardt Datsomor](https://www.linkedin.com/in/gerhardt-datsomor/)
8. [John Ayernor](https://www.linkedin.com/in/john-kwabena-ayernor-45b497186/)

## Contact

For questions or comments, please email [natural.language.processing.gh@gmail.com](mailto:natural.language.processing.gh@gmail.com).

To submit your contributions, send them to [michsethowusu@gmail.com](mailto:michsethowusu@gmail.com).

## License

This is an open community project. We welcome researchers, developers, and language enthusiasts to participate and help advance NLP for Ghanaian languages.
