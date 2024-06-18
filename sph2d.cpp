/* sph2d.cpp: ����Ӧ�ó������ڵ㡣�Ѹ���ΪULPH����
*��Դ����copyһ�ݣ�����ʹ��cmake.Ŀǰ�����ѿ������ѡ��debug���Ǵ��У�������ϵ���е���
* ---------------------------
* 1��NS����Ϊ̬��ʽ��
* 2�������߽�Ϊ���ڱ߽磬�ҽ����������ѽ��
* 3������������Ӻ�ɢ����
* 4��������
* 5��single_step_temperature_gaojie�����ˣ����ǵ����漰�¶�ʱ��single_step�͹��ˣ�����single_step��û�ġ�
* 6����single_step,�Ѿ���������
* ����Ϊ��Couette flow
*		 ���Ժ����价�����������Ϊ��׼��
*/
#include <iostream>
//#include "sph2d.h"
#include "domain.h"
#include "particle.h"
#include"omp.h"

using namespace std;

int main()
{
	cout << "************\033[90m time: 2023/6/16 \33[m************" << endl;
	cout << "************\033[90m Tube bank heat exchanger(one tube) \33[m************" << endl;
	//	int num = omp_get_num_procs();//���Բ���
	//	cout << num << endl;//��ô���������
	/*#pragma omp parallel
		{
			cout << "Test" << endl;
		}*/
	const double spacing = 0.005;// 0.005
	const double width = 0.2;//��0.05
	const double height = 0.09;//��,��Ҫ�߽����һ��߽�����������
	const double inletWidth = 4 * spacing;//(���ڱ߽��)������
	const double inletDiameter = height;//��
	std::cout << "----------ģ�ͳߴ�----------- " << std::endl;
	std::cout << "----width =  " << -inletWidth << "-" << width  << std::endl;
	std::cout << "----height =  " << -height*0.5 << "-" << height * 0.5  << std::endl;
	std::cout << "----��������� = " << inletWidth << std::endl;//��������Ȳ����ڸ����ʼ�ٶȵ�������
	std::cout << "----��������  = " << width + spacing * 0.5 << "-" <<  width + spacing * 4.5 << std::endl;
	std::cout << "----��������  = " << -spacing * (int)(inletWidth / spacing) << "-" <<  - spacing * (int(inletWidth / spacing) - 4.0) << std::endl;
	/*std::cout << "----�������ٶ�  = 1.0 " << std::endl;
	std::cout << "----Բ���뾶  = 0.1 " << std::endl;
	std::cout << "----Բ��λ��  = ��0.4��0�� " << std::endl;
	std::cout << "----��ŵ��Re  = rho * vel * d / visco ="  << (double)(1000*1.0*0.2/ 0.001) << std::endl;*/
	std::cout << "----���Ӽ��  = " << spacing << std::endl;
	std::cout << "----------------------------- " << std::endl;
	sph::domain* dm = new sph::domain();
	dm->setp_back(0);
	dm->usingProcessbar(true);//�����������
	dm->setParallel(true);//�ڹ��캯����ʼ���У�Parallel=false
	//dm->readinVel("cough_velocity.csv");
	dm->createDomain1(spacing, width, height, inletWidth, inletDiameter);
	//dm->createDomain(0.0025,0.05,0.1,0.2,0.02);
	dm->setOutScreen(10);
	dm->setOutIntval(1000);
	dm->setnbskin(0.1);//�����ڲ�����drmax + drmax2 <= neibskin * dp * 1.01 
	dm->setRatio(0.00);
	dm->setInitVel(0);//��ʼ�ٶȣ������зǱ߽��particle���ӣ�no ghost��
	dm->setShifting(sph::ShiftingType::None);//����run�е�shifting�У���̬�ȴ�������ҪShifting
	dm->setShiftingCoe(1.0);//��setShiftingһ������run�е�shifting��
	dm->setOutlet(width - 3.5 * spacing, sph::Direction::Right, 4);//���������߽磬������outlet����(Buffer����)
	dm->writeInitial();//д���ʼ�ļ���"initial.dat"
	dm->solve(10000);
	return 0;
}
