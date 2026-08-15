# Laboratórios de Deep Learning | Capacitação UFCG

## Estrutura

| Lab | Tema |
|-----|------|
| `Lab1` | Configuração do ambiente |
| `Lab2` | Redes Recorrentes (RNN / LSTM) |
| `Lab3` | Autoencoders |
| `Lab4` | Transformers |
| `Lab5` | Redes Generativas Adversariais (GANs) |

## `Lab1` - Configuração do Ambiente

Documentação de instalação e configuração do ambiente de desenvolvimento com PyTorch e TensorFlow, incluindo verificação de suporte a GPU e validação das dependências necessárias para os laboratórios seguintes.

## `Lab2` - Redes Recorentes (RNN / LSTM)

Introdução às redes recorrentes para processamento de sequências de texto. O laboratório implementa em PyTorch tanto uma RNN simples quanto uma LSTM, aplicadas à tarefa de análise de sentimentos no IMDB. O ciclo de treinamento é implementado manualmente, sem uso de abstrações de alto nível.

## `Lab3` - Autocodificadores

Exploração da arquitetura encoder-decoder aplicada à colorização de paisagens. A pt1 aborda o pipeline em espaço RGB, em que o modelo recebe uma imagem em escala de cinza e reconstrói a versão colorida. A pt2 migra para o espaço de cor CIELAB, separando luminosidade de cromaticidade, o que torna a tarefa de aprendizado mais estruturada. A pt3 muda de domínio e aplica um autoencoder convolucional 1D a sinais de ECG, explorando a detecção de anomalias cardíacas.

## `Lab4` - Transformers

Estudo da arquitetura Transformer aplicada à mesma tarefa explorada no Lab2, permitindo comparação direta de desempenho. A pt1 constrói o modelo do zero em PyTorch, implementando embedding posicional e atenção multi-head de forma explícita. A pt2 realiza fine-tuning do modelo BERT pré-treinado, explorando a transferência de conhecimento adquirido em larga escala para a tarefa específica de classificação de sentimentos.  O notebook de Vision Transformers estende o mecanismo de atenção para o domínio visual, na qual uma imagem é dividida em patches tratados como tokens de uma sequência, aplicado à classificação do CIFAR-10.

## `Lab5` - Redes Generativas Adversariais (GANs)

Aplicação de redes generativas adversariais à tarefa de colorização, dando continuidade ao lab3. O laboratório implementa a arquitetura Pix2Pix, composta por um Gerador no formato U-Net e um Discriminador PatchGAN condicional. O Gerador recebe o canal de luminosidade L e produz os canais de cor AB, enquanto o Discriminador avalia a coerência local da colorização por regiões da imagem, forçando resultados mais saturados e semanticamente consistentes do que os obtidos com a abordagem de regressão do Lab3.

## Aplicações de Continuidade

### Análise de Sentimentos (IMDB)

Os laboratórios 2 e 4 desenvolvem modelos para a mesma tarefa, sendo a classificação binária de sentimento no dataset Stanford IMDB (50 000 resenhas, positivo/negativo). Os três modelos compartilham os mesmos parâmetros de dados para permitir comparação justa via matriz de confusão.

```
Lab2  ->  RNN / LSTM                     (arquitetura recorrente, sequencial)
Lab4  ->  Transformer (do zero)          (atenção multi-cabeça, posicional)
Lab4  ->  BERT fine-tuning               (bert-base-uncased, pré-treinado)
```

Explorado desde o processamento sequencial estrito (RNN) à atenção paralela (Transformer) e ao aproveitamento de representações pré-treinadas em escala (BERT).

### Colorização de Paisagens

Os laboratórios 3 e 5 abordam a mesma tarefa, sendo colorização de imagens em escala de cinza usando o dataset Landscapes (~4 000 imagens).

```
lab3-pt1  ->  U-Net + MSE               (regressão pixel a pixel, espaço RGB)
lab3-pt2  ->  U-Net + MSE               (regressão pixel a pixel, espaço CIELAB)
lab5      ->  Pix2Pix (U-Net + PatchGAN) (adversarial + L1, GAN condicional)
```

A pt1 aborda o problema diretamente no espaço RGB. A pt2 migra para o espaço CIELAB, que separa luminosidade de cromaticidade e torna o aprendizado mais estruturado — o modelo recebe o canal L como entrada e prevê os canais AB como saída. Ainda assim, a U-Net treinada com MSE converge para a média das cores plausíveis, produzindo imagens dessaturadas. O Pix2Pix do Lab5 resolve isso introduzindo um Discriminador PatchGAN que força o Gerador a produzir cores semanticamente coerentes e com maior saturação.

## Referências principais

- Hochreiter & Schmidhuber (1997). *Long Short-Term Memory.* Neural Computation.
- Vaswani et al. (2017). *Attention Is All You Need.* NeurIPS.
- Devlin et al. (2018). *BERT: Pre-training of Deep Bidirectional Transformers.* NAACL.
- Ronneberger et al. (2015). *U-Net: Convolutional Networks for Biomedical Image Segmentation.* MICCAI.
- Isola et al. (2017). *Image-to-Image Translation with Conditional Adversarial Networks.* CVPR.
