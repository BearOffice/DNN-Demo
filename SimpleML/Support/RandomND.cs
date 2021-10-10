using System;
using System.Collections.Generic;

namespace SimpleML.Support
{
    public class RandomND
    {
        private readonly Random _random;
        private readonly double _avg;
        private readonly double _sd;
        private readonly Queue<double> _numQueue = new();

        public RandomND(double avg, double sd)
        {
            _random = new Random();
            _avg = avg;
            _sd = sd;
        }

        public RandomND(int randomseed, double avg, double sd)
        {
            _random = new Random(randomseed);
            _avg = avg;
            _sd = sd;
        }

        public double Next()
        {
            if (_numQueue.Count == 0)
                AddTwoNums();

            return _numQueue.Dequeue();
        }

        private void AddTwoNums()
        {
            double x, y;
            double z1, z2;

            x = _random.NextDouble();
            y = _random.NextDouble();

            z1 = _sd * Math.Sqrt(-2.0 * Math.Log(x)) * Math.Cos(2.0 * Math.PI * y) + _avg;
            z2 = _sd * Math.Sqrt(-2.0 * Math.Log(x)) * Math.Sin(2.0 * Math.PI * y) + _avg;

            _numQueue.Enqueue(z1);
            _numQueue.Enqueue(z2);
        }
    }
}
