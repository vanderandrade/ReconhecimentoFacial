#include<iostream>
#include<sstream>
#include<string>
#include "opencv2/opencv.hpp"

using std::cout;
using std::string;
using std::stringstream;
using std::vector;

using namespace cv;

std::string to_string(int i)
{
    std::stringstream ss;
    ss << i;
    return ss.str();
}

string caminhoImagem (int i)
{
	string caminhoAbs, caminho, nomeImagem;

	caminho = "/home/vander/ProjetosOpenCV/capFrame/imagens/";

	nomeImagem = "img";
    nomeImagem += to_string(i);
    nomeImagem += ".pgm";
    
    caminhoAbs = caminho;
    caminhoAbs += nomeImagem;

    return caminhoAbs;
} 

int main(int argc, char** argv)
{
	int i = 0;
	string caminho;
	vector<int> ParametrosCompressao;
	
    VideoCapture cap;
    if(!cap.open(0))
        return -1;

    ParametrosCompressao.push_back(CV_IMWRITE_PXM_BINARY);
    ParametrosCompressao.push_back(0);  // tipo do PXM, ou seja, pgm

    for(;;)
    {
    	Mat frame;

    	cap >> frame;
        if(frame.empty())
        	break;

        imshow("this is you, smile! :)", frame);

        if(waitKey(10) == 32) //Esp
        {
        	caminho = caminhoImagem(i);
        	imwrite(caminho, frame, ParametrosCompressao);
        	i++;
        }

        if(i == 10)
        	break;
	}

    return 0;
}
