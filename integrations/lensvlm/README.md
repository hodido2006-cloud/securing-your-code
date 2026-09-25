# LensVLM-9B integration

Client for [apple/LensVLM-9B](https://huggingface.co/apple/LensVLM-9B), a 9B vision-language
model (fine-tuned from Qwen3.5-9B) that answers questions about long documents. It renders the
text as compressed page images (5x / 10x / 15x) and expands only the pages it needs.

## Why a hosted Space and not local inference

The weights are one 18.8 GB `model.safetensors` file and need a GPU. This client calls the
hosted demo Space [`hugging-apps/lensvlm-9b-demo`](https://hf.co/spaces/hugging-apps/lensvlm-9b-demo)
through `gradio_client`. To run the model on your own GPU, use Apple's reference code:
<https://github.com/apple-aiml-research/ml-lensvlm>.

## Run it

```bash
cd integrations/lensvlm
pip install -r requirements.txt
export HF_TOKEN=hf_...   # optional; reduces anonymous rate limiting / queueing

python lensvlm_client.py --file sample_contract.txt \
    --question "What is the retention cap and the delay damages cap?" --compression 5x
```

If the call fails with an endpoint error, run `python lensvlm_client.py --list-api`
to print the Space's current endpoints, then pass the right one with `--api-name`.

## Limits

- **Text only.** The input is plain text. Convert PDFs or Word files to `.txt` first.
- **Licence.** Apple Machine Learning Research Model License: research use, not commercial use.
  Check it before you use the model on live project documents.
- **Confidentiality.** The document goes to a public Hugging Face Space. Do not send confidential
  contract or pricing data.
- **Accuracy.** Check the answers against the source clause before you rely on them.
