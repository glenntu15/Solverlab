#pragma once
class Timers
{
public:
	static Timers& getInstance()
	{
		static Timers instance;
		return instance;
	}

	~Timers();

	double second(void);

};

