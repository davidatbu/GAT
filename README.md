# Requirements
 - `dev-requirements.txt` indicates packages necessary for testing, linting,
    autoformatting. Not required to use the package.
 -  `requirements.txt` is for machines with GPUs.
 -  `requirements_no_gpu.txt` is for machines without GPUs.
  
# TODO: Write an actual README.

You will need to do the following also:

	spacy download en_core_web_sm


To use SRL, I _think_ `allennlp` uses the "_md" version to do tokenization.

	spacy download en_core_web_md

PyDot to visualize
