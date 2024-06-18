/* sph2d.cpp: 定义应用程序的入口点。已更改为ULPH程序。
*将源程序copy一份，不再使用cmake.目前并行已开，如果选用debug就是串行，方便设断点进行调试
* ---------------------------
* 1、NS方程为态形式的
* 2、进出边界为周期边界，且建近邻问题已解决
* 3、能量方程添加耗散函数
* 4、加湍流
* 5、single_step_temperature_gaojie改完了，但是当不涉及温度时，single_step就够了；但是single_step还没改。
* 6、改single_step,已经加上湍流
* 算例为：Couette flow
*		 ！以后新配环境都用这个作为基准！
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
	//	int num = omp_get_num_procs();//测试并行
	//	cout << num << endl;//获得处理器个数
	/*#pragma omp parallel
		{
			cout << "Test" << endl;
		}*/
	const double spacing = 0.005;// 0.005
	const double width = 0.2;//长0.05
	const double height = 0.09;//高,需要边界各加一层边界粒子来考虑
	const double inletWidth = 4 * spacing;//(周期边界的)流入区
	const double inletDiameter = height;//高
	std::cout << "----------模型尺寸----------- " << std::endl;
	std::cout << "----width =  " << -inletWidth << "-" << width  << std::endl;
	std::cout << "----height =  " << -height*0.5 << "-" << height * 0.5  << std::endl;
	std::cout << "----流入区宽度 = " << inletWidth << std::endl;//流入区宽度不等于赋予初始速度的区域宽度
	std::cout << "----流出区域  = " << width + spacing * 0.5 << "-" <<  width + spacing * 4.5 << std::endl;
	std::cout << "----流入区域  = " << -spacing * (int)(inletWidth / spacing) << "-" <<  - spacing * (int(inletWidth / spacing) - 4.0) << std::endl;
	/*std::cout << "----流入区速度  = 1.0 " << std::endl;
	std::cout << "----圆柱半径  = 0.1 " << std::endl;
	std::cout << "----圆心位置  = （0.4，0） " << std::endl;
	std::cout << "----雷诺数Re  = rho * vel * d / visco ="  << (double)(1000*1.0*0.2/ 0.001) << std::endl;*/
	std::cout << "----粒子间距  = " << spacing << std::endl;
	std::cout << "----------------------------- " << std::endl;
	sph::domain* dm = new sph::domain();
	dm->setp_back(0);
	dm->usingProcessbar(true);//输出界面设置
	dm->setParallel(true);//在构造函数初始化中，Parallel=false
	//dm->readinVel("cough_velocity.csv");
	dm->createDomain1(spacing, width, height, inletWidth, inletDiameter);
	//dm->createDomain(0.0025,0.05,0.1,0.2,0.02);
	dm->setOutScreen(10);
	dm->setOutIntval(1000);
	dm->setnbskin(0.1);//建近邻参数：drmax + drmax2 <= neibskin * dp * 1.01 
	dm->setRatio(0.00);
	dm->setInitVel(0);//初始速度，给所有非边界的particle粒子（no ghost）
	dm->setShifting(sph::ShiftingType::None);//用于run中的shifting中，静态热传导不需要Shifting
	dm->setShiftingCoe(1.0);//与setShifting一样用于run中的shifting中
	dm->setOutlet(width - 3.5 * spacing, sph::Direction::Right, 4);//设置流出边界，会生成outlet粒子(Buffer粒子)
	dm->writeInitial();//写入初始文件："initial.dat"
	dm->solve(10000);
	return 0;
}
