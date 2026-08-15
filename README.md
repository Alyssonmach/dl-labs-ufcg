# Laboratórios de Deep Learning | Capacitação UFCG

---

## Estrutura

| Lab | Tema |
|-----|------|
| `lab1` | Configuração do ambiente |
| `lab2` | Redes Recorrentes (RNN / LSTM) |
| `lab3` | Autoencoders |
| `lab4` | Transformers |
| `lab5` | Redes Generativas Adversariais (GANs) |

---

## Análise de Sentimentos (IMDB)

Os laboratórios 2 e 4 desenvolvem modelos para a mesma tarefa: classificação binária de sentimento no dataset Stanford IMDB (50 000 resenhas, positivo/negativo).  

Os três modelos compartilham os mesmos parâmetros de dados para permitir comparação justa via matriz de confusão.

```
lab2  ->  RNN / LSTM                     (arquitetura recorrente, sequencial)
lab4  ->  Transformer (do zero)          (atenção multi-cabeça, posicional)
lab4  ->  BERT fine-tuning               (bert-base-uncased, pré-treinado)
```

Explorado desde o processamento sequencial estrito (RNN) à atenção paralela (Transformer) e ao aproveitamento de representações pré-treinadas em escala (BERT).

---

## Colorização de Paisagens

Os laboratórios 3 e 5 abordam a mesma tarefa, sendo colorização de imagens em escala de cinza usando o dataset Landscapes (~4 000 imagens).  

Ambos operam no espaço de cor CIELAB, recebendo o canal de luminosidade L como entrada e prevendo os canais de cromaticidade AB como saída.

```
lab3-pt2  ->  U-Net + MSE (L1)           (regressão pixel a pixel, espaço CIELAB)
lab5      ->  Pix2Pix (U-Net + PatchGAN) (adversarial + L1, GAN condicional)
```

A U-Net treinada com MSE converge para a média das cores plausíveis, produzindo imagens dessaturadas (*efeito sépia*). O Pix2Pix introduz um Discriminador PatchGAN que força o Gerador a produzir cores semanticamente coerentes e com maior saturação.

---

## Detalhes por laboratório

### `lab1` — Configuração do Ambiente

Documentação de instalação e configuração do ambiente de desenvolvimento com PyTorch e TensorFlow, incluindo verificação de suporte a GPU e validação das dependências necessárias para os laboratórios seguintes.

### `lab2` — RNN / LSTM

Introdução às redes recorrentes para processamento de sequências de texto. O laboratório implementa em PyTorch tanto uma RNN simples quanto uma LSTM, aplicadas à tarefa de análise de sentimentos no IMDB. O ciclo de treinamento é implementado manualmente, sem uso de abstrações de alto nível, tornando explícitas as etapas de forward pass, cálculo de gradientes e atualização de pesos.

### `lab3` — Autoencoders

Exploração da arquitetura encoder-decoder aplicada à colorização de paisagens. A pt1 aborda o pipeline em espaço RGB, em que o modelo recebe uma imagem em escala de cinza e reconstrói a versão colorida. A pt2 migra para o espaço de cor CIELAB, separando luminosidade de cromaticidade, o que torna a tarefa de aprendizado mais estruturada. A pt3 estende o conceito para autoencoders variacionais (VAE), introduzindo um espaço latente probabilístico.

### `lab4` — Transformers

Estudo da arquitetura Transformer aplicada à mesma tarefa do lab2, permitindo comparação direta de desempenho. A pt1 constrói o modelo do zero em PyTorch, implementando embedding posicional e atenção multi-cabeça de forma explícita. A pt2 realiza fine-tuning do modelo BERT pré-treinado, explorando a transferência de conhecimento adquirido em larga escala para a tarefa específica de classificação de sentimentos.

### `lab5` — GANs

Aplicação de redes generativas adversariais à tarefa de colorização, dando continuidade ao lab3. O laboratório implementa a arquitetura Pix2Pix, composta por um Gerador no formato U-Net e um Discriminador PatchGAN condicional. O Gerador recebe o canal de luminosidade L e produz os canais de cor AB, enquanto o Discriminador avalia a coerência local da colorização por regiões da imagem, forçando resultados mais saturados e semanticamente consistentes do que os obtidos com a abordagem de regressão do lab3.

---

## Referências principais

- Hochreiter & Schmidhuber (1997). *Long Short-Term Memory.* Neural Computation.
- Vaswani et al. (2017). *Attention Is All You Need.* NeurIPS.
- Devlin et al. (2018). *BERT: Pre-training of Deep Bidirectional Transformers.* NAACL.
- Ronneberger et al. (2015). *U-Net: Convolutional Networks for Biomedical Image Segmentation.* MICCAI.
- Isola et al. (2017). *Image-to-Image Translation with Conditional Adversarial Networks.* CVPR.
