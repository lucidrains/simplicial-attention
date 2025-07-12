<img src="./fig2.png" width="400px"></img>

## Simplicial Attention - (wip)

Implementation of [2-simplicial attention](https://arxiv.org/abs/1909.00668) proposed by Clift et al. (2019) and the recent attempt to make practical in [Fast and Simplex](https://arxiv.org/abs/2507.02754), Roy et al. (2025)

[Paper explanation by Gabriel Mongaras](https://www.youtube.com/watch?v=W-0LSbTnbVc)

## Contributing

First install with `pytest`

```shell
$ pip install '.[test]'
```

Then add your code and make sure it passes

```shell
$ pytest tests
```

## Citations

```bibtex
@misc{roy2025fastsimplex2simplicialattention,
    title   = {Fast and Simplex: 2-Simplicial Attention in Triton}, 
    author  = {Aurko Roy and Timothy Chou and Sai Surya Duvvuri and Sijia Chen and Jiecao Yu and Xiaodong Wang and Manzil Zaheer and Rohan Anil},
    year    = {2025},
    eprint  = {2507.02754},
    archivePrefix = {arXiv},
    primaryClass = {cs.LG},
    url     = {https://arxiv.org/abs/2507.02754}, 
}
```

```bibtex
@misc{clift2019logic2simplicialtransformer,
    title   = {Logic and the $2$-Simplicial Transformer}, 
    author  = {James Clift and Dmitry Doryn and Daniel Murfet and James Wallbridge},
    year    = {2019},
    eprint  = {1909.00668},
    archivePrefix = {arXiv},
    primaryClass = {cs.LG},
    url     = {https://arxiv.org/abs/1909.00668}, 
}
```

```bibtex
@article{Peng2024OnLO,
    title     = {On Limitations of the Transformer Architecture},
    author    = {Binghui Peng and Srini Narayanan and Christos Papadimitriou},
    journal   = {ArXiv},
    year      = {2024},
    volume    = {abs/2402.08164},
    url       = {https://api.semanticscholar.org/CorpusID:267636545}
}
```

