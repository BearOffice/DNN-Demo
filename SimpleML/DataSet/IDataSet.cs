using System;
using SimpleMath.Collections;

namespace SimpleML.DataSet
{
    public interface IDataSet
    {
        public (DoubleMatrix, string[]) GetDataSet();
    }
}
