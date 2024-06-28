#pragma once

namespace sph {
	enum class BoundaryType    //0为流体域，1为边界域
	{
		Bulk = 0,
		Boundary = 1
	};

	enum class BoundaryConditionType
	{
		FreeSlip = 1,
		NoSlip = 2
	};

	enum class FixType
	{
		Free = 0,
		Fixed = 1,
		Moving = 2
	};

	enum class InoutType
	{
		Fluid = 0,
		Inlet = 1,
		Outlet = 2,
		Buffer = 3,
		Ghost = 4
	};

	enum class FluidType
	{
		Air,
		Water,
		Moisture
	};
}

constexpr auto MAX_NEIB = 50;