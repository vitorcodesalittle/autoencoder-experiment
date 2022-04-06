Um pequeno roadmap para isso ficar entregável:
- Treinamento de modelos mais uniforme
  - Treinar autoencoders para usar mesmos hiperparâmetros
  - Agilizar treinamento do classifier from coder ( e melhorá-lo )
- Comparação:
  - Avaliação nos dados de teste (acurácia e loss)
  - Avaliação da perda durante o treinamento
- Modelos:
  - Autoencoder Linear 1024 x AutoecoderLinear 512 x AutoencoderLinear 256 x AutoencoderLinear 64
  - Autoencoder Conv 64 x AutoencoderConv 32 x AutoencoderConv 16 x AutoencoderConv 8
  - Classificadores_Autoencoder_{Linear,Conv} X entre si

- Documentar tomadas de decisão

