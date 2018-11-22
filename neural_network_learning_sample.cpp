#include "neural_network_learning_sample.hpp"

NeuralNetworkLearningSample::NeuralNetworkLearningSample()
{
    input = Q_NULLPTR;
    target = Q_NULLPTR;
    inputSize = targetSize = 0;
}

NeuralNetworkLearningSample::NeuralNetworkLearningSample(
        const NeuralNetworkLearningSample& sample )
    : NeuralNetworkLearningSample()
{
    setInputVectorSize( sample.inputSize );
    setTargetVectorSize( sample.targetSize );
    for( quint32 i = 0; i < inputSize; i++ )
    {
        input[ i ] = sample.input[ i ];
    }
    for( quint32 i = 0; i < targetSize; i++ )
    {
        target[ i ] = sample.target[ i ];
    }
    mark = sample.mark;
}

NeuralNetworkLearningSample::NeuralNetworkLearningSample(
        NeuralNetworkLearningSample&& sample )
    : NeuralNetworkLearningSample()
{
    input = sample.input;
    target = sample.target;
    inputSize = sample.inputSize;
    targetSize = sample.targetSize;
    mark = std::move( sample.mark );

    sample.input = Q_NULLPTR;
    sample.target = Q_NULLPTR;
    sample.inputSize = sample.targetSize = 0;
    sample.mark = QString();
}

NeuralNetworkLearningSample::~NeuralNetworkLearningSample()
{
    if( input ) delete[] input;
    if( target ) delete[] target;
}

NeuralNetworkLearningSample& NeuralNetworkLearningSample::operator=(
        const NeuralNetworkLearningSample& sample )
{
    setInputVectorSize( sample.inputSize );
    setTargetVectorSize( sample.targetSize );
    for( quint32 i = 0; i < inputSize; i++ )
    {
        input[ i ] = sample.input[ i ];
    }
    for( quint32 i = 0; i < targetSize; i++ )
    {
        target[ i ] = sample.target[ i ];
    }
    mark = sample.mark;
    return *this;
}

NeuralNetworkLearningSample& NeuralNetworkLearningSample::operator=(
        NeuralNetworkLearningSample&& sample )
{
    if( input ) delete[] input;
    if( target ) delete[] target;

    input = sample.input;
    target = sample.target;
    inputSize = sample.inputSize;
    targetSize = sample.targetSize;
    mark = std::move( sample.mark );

    sample.input = Q_NULLPTR;
    sample.target = Q_NULLPTR;
    sample.inputSize = sample.targetSize = 0;
    sample.mark = QString();
    return *this;
}

void NeuralNetworkLearningSample::setInputVector( uchar* data, const quint32 size )
{
    if( input ) delete[] input;
    inputSize = size;
    input = data;
}

void NeuralNetworkLearningSample::setTargetVector( uchar* data, const quint32 size )
{
    if( target ) delete[] target;
    targetSize = size;
    target = data;
}

uchar* NeuralNetworkLearningSample::getInputVector() const
{
    return input;
}

uchar* NeuralNetworkLearningSample::getTargetVector() const
{
    return target;
}

void NeuralNetworkLearningSample::setInputVectorSize( const quint32 size )
{
    if( input ) delete[] input;
    inputSize = size;
    input = new uchar[ size ];
}

void NeuralNetworkLearningSample::setTargetVectorSize( const quint32 size )
{
    if( target ) delete[] target;
    targetSize = size;
    target = new uchar[ size ];
}

quint32 NeuralNetworkLearningSample::getInputVectorSize() const
{
    return inputSize;
}

quint32 NeuralNetworkLearningSample::getTargetVectorSize() const
{
    return targetSize;
}

void NeuralNetworkLearningSample::setInputValue(
        const quint32 index, const uchar value )
{
    if( index < inputSize )
    {
        input[ index ] = value;
    }
}

void NeuralNetworkLearningSample::setTargetValue(
        const quint32 index, const uchar value )
{
    if( index < targetSize )
    {
        target[ index ] = value;
    }
}

uchar NeuralNetworkLearningSample::getInputValue( const quint32 index ) const
{
    if(  index >= inputSize ) return 0;
    return input[ index ];
}

uchar NeuralNetworkLearningSample::getTargetValue( const quint32 index ) const
{
    if(  index >= targetSize ) return 0;
    return target[ index ];
}

void NeuralNetworkLearningSample::setMark( const QString& mark )
{
    this->mark = mark;
}

const QString& NeuralNetworkLearningSample::getMark() const
{
    return mark;
}
