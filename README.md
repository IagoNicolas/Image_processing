# Processamento de imagem

F. Processamento de imagem
Nesse projeto, é apresentado uma introdução ao processamento de imagem utilizando ferramenta computacional. Para a realização desse projeto, considere o módulo adicional do Scilab scicv, Python com OpenCV ou qualquer outra linguagem com processamento de imagem. Uma imagem qualquer pode ser convertida em um sinal do tipo x[m, n], em que m = 0, 1, . . . , M−1 e n = 0, 1, . . . , N−1, imagem com resolução N × M. O valor x[m, n] representa a intensidade de luz (entre 0 e 1) caso a imagem seja monocromática (preto e branco). Se a imagem é colorida, a imagem é representada por x[m, n, c], com c = 0, 1, 2 ou c = R, G, B, representando a intensidade do vermelho, verde e azul respectivamente em cada componente (RGB).

1) Faça a aquisição de uma imagem e converta colorida e
uma preto e branco (pode converter a imagem colorida
para monocromática).
2) Implemente uma função que aplica um filtro de linha,
H`(z), em tempo discreto para cada linha x[m, :] e
em seguida (a partir do resultado anterior), aplique um
filtro de coluna, Hc(z), para cada coluna x[:, n] (isso é
equivalente a um filtro 2D separável).
3) Utilize um mesmo derivador em tempo discreto como
H`(z) e Hc(z), projete o derivador utilizando uma
aproximação do derivador ideal com pelo menos 16
pontos de resposta ao impulso com fase linear. mostre
o resultado na imagem monocromática e colorida.
4) Considere agora a aplicação de um filtro passa todas com
polos em ±1/2, ±1/4 e ±3/4 na imagem, mostre e explique
o resultado (avalie o efeito da fase não linear para processamento de imagem). Mostre o resultado na imagem monocromática e colorida para cada polo. Comente os casos com maior/menor distorção da imagem. 10.23 e 10.30
5) Implemente um sistema que detecta bordas
monocromáticas e bordas de uma determinada cor
(exemplo: borda azul) e teste aplicando suas imagens.

# Todo

Questão 1: Done.

Questão 2: Adicionar os seguintes filtros (from scratch): (filtro pronto?)
* Passa baixa;
* Passa alta;
* Passa banda;
* Passa faixa;
* Rejeita banda;
* Rejeita faixa;
* Moving average;
* Savgol;
* ???.

Questão 3: Projetar o derivador e implementar.

Questão 4: Projetar e implementar.

Questão 5: Canny edge detection reescrito, pendente integração.

# Code of conduct
https://www.gnu.org/philosophy/kind-communication.en.html