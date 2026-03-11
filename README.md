# Neural Storyteller

> Image captioning with a CNN encoder and LSTM decoder, trained on Flickr30k.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![Dataset](https://img.shields.io/badge/Dataset-Flickr30k-9B72CB?style=flat-square)

---

## Overview

Neural Storyteller is an end-to-end image captioning system built with PyTorch and deployed as a Streamlit web app. Upload any image and the model generates a natural-language description of its contents in real time.

The architecture follows the classic **Show and Tell** paradigm — a convolutional encoder extracts visual features from the image, and a recurrent decoder generates a caption token by token using beam search.

---

## Architecture

```
Image → ResNet-50 → Linear (2048 → 512) → LSTM Decoder → Caption
```

| Component | Detail |
|---|---|
| Feature Extractor | ResNet-50 (pretrained, final layer removed) |
| Projection | Linear layer, 2048 → 512 dimensions |
| Decoder | Single-layer LSTM with learned word embeddings |
| Vocabulary | Built from Flickr30k captions with frequency thresholding |
| Decoding | Beam search (configurable beam width 1–5) |
| Training Dataset | Flickr30k (~31,000 images, 5 captions each) |

---

## Demo

The app is built with Streamlit and features a Gemini-inspired dark UI.

- Upload a PNG, JPG, or JPEG image
- Adjust beam width and max token length from the sidebar
- Click **Generate Caption** to run inference
- Download the output as a `.txt` file

---

## Project Structure

```
neural-storyteller/
├── app.py                     # Streamlit application
├── train.py                   # Training loop
├── dataset.py                 # Flickr30k dataset loader and transforms
├── model.py                   # EncoderCNN, DecoderRNN, Seq2Seq definitions
├── vocab.py                   # Vocabulary builder
├── vocab.pkl                  # Serialized vocabulary (generated after training)
├── best_flickr30k_model.pth   # Saved model weights (generated after training)
└── requirements.txt
```

---

## Setup

**1. Clone the repository**

```bash
git clone https://github.com/your-username/neural-storyteller.git
cd neural-storyteller
```

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

**3. Download the Flickr30k dataset**

Place images in `data/flickr30k_images/` and the captions CSV at `data/captions.csv`.

**4. Train the model**

```bash
python train.py
```

This will generate `vocab.pkl` and save the best checkpoint to `best_flickr30k_model.pth`.

**5. Run the app**

```bash
streamlit run app.py
```

---

## Requirements

```
torch
torchvision
streamlit
Pillow
```

---

## How It Works

1. **Encoding** — The input image is resized to 224×224 and passed through ResNet-50. The final 2048-dimensional pooled feature vector is projected to 512 dimensions via a learned linear layer with ReLU activation.

2. **Decoding** — The encoded feature vector initialises the LSTM decoder. At each timestep, the decoder attends to the previous token embedding and its hidden state to predict the next word from the vocabulary.

3. **Beam Search** — Rather than greedily picking the single highest-probability token at each step, beam search maintains the top *k* candidate sequences simultaneously, yielding more fluent and coherent captions.

---

## Results

The model was trained from scratch on Flickr30k for 10 epochs with the Adam optimizer and cross-entropy loss. Representative outputs:

| Image Description | Generated Caption |
|---|---|
| Dog running on beach | *a dog is running on the beach* |
| Group of people at a table | *a group of people sitting around a table* |
| Child on a bicycle | *a young boy is riding a bicycle* |

---

## Acknowledgements

- [Flickr30k Dataset](http://shannon.cs.illinois.edu/DenotationGraph/)
- [Show and Tell: A Neural Image Caption Generator](https://arxiv.org/abs/1411.4555) — Vinyals et al., 2014
- [PyTorch](https://pytorch.org/) and [Streamlit](https://streamlit.io/)

---

## License

MIT