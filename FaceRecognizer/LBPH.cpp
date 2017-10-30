#include "opencv2/core.hpp"
#include "opencv2/face.hpp"
#include "opencv2/highgui.hpp"

#include <iostream>
#include <fstream>
#include <sstream>

using namespace cv;
using namespace cv::face;
using namespace std;

static void lerArquivoCSV(const string& nomeArquivo, vector<Mat>& imagens, vector<int>& ids, char separador = ';') {
	std::ifstream arquivo(nomeArquivo.c_str(), ifstream::in);
	
	if (!arquivo)
		CV_Error(Error::StsBadArg, "O arquivo oferecido não é válido!");

	string linha, caminho, id;
	
	while (getline(arquivo, linha)) {
		stringstream linhaAux(linha);
		
		getline(linhaAux, caminho, separador);
		getline(linhaAux, id);
		if(!caminho.empty() && !id.empty()) {
			imagens.push_back(imread(caminho, 0));
			ids.push_back(atoi(id.c_str()));
		}
	}
}

int reconhecimentoFacial(string arquivoBaseCSV, Mat imagemCapturada) {
	vector<Mat> imagens;
	vector<int> ids;

	try {
		lerArquivoCSV(arquivoBaseCSV, imagens, ids);
	} catch (cv::Exception& e) {
		cerr << "Erro ao abrir o arquivo \"" << arquivoBaseCSV << "\". Motivo: " << e.msg << endl;
		exit(1);
	}

	Ptr<LBPHFaceRecognizer> modelo = LBPHFaceRecognizer::create();
	modelo->train(imagens, ids);

	int idClassidicado = -1;
	double confianca = 0.0;
	modelo->predict(imagemCapturada, idClassidicado, confianca);

	cout << "Confiança: " << confianca << endl;

	if(confianca > 70.0)
		return -1;
	return idClassidicado;
}

int main(int argc, char** argv)
{
    if(argc != 3)
    {
        cout << "chamada: " << argv[0] << " <baseCSV> <imagem>" << endl;
        return -1;
    }

    Mat imagem;
    imagem = imread(argv[2], 0);

    if(!imagem.data)
    {
        printf("Caminho de imagem inválido!\n");
        return -1;
    }
    
    cout << "Classe: " << reconhecimentoFacial(argv[1], imagem) << endl;

    return 0;
}