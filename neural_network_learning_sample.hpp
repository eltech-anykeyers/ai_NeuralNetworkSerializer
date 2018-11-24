#ifndef NEURAL_NETWORK_LEARNING_SAMPLE_HPP
#define NEURAL_NETWORK_LEARNING_SAMPLE_HPP

#include <QString>

class NeuralNetworkLearningSample
{
public:
    NeuralNetworkLearningSample();
    NeuralNetworkLearningSample( const NeuralNetworkLearningSample& sample );
    NeuralNetworkLearningSample( NeuralNetworkLearningSample&& sample );
    ~NeuralNetworkLearningSample();

    NeuralNetworkLearningSample& operator=(
            const NeuralNetworkLearningSample& sample );
    NeuralNetworkLearningSample& operator=(
            NeuralNetworkLearningSample&& sample );

    void setInputVector( uchar* data, const quint32 size );
    void setTargetVector( uchar* data, const quint32 size );
    uchar* getInputVector() const;
    uchar* getTargetVector() const;

    void setInputVectorSize( const quint32 size );
    void setTargetVectorSize( const quint32 size );
    quint32 getInputVectorSize() const;
    quint32 getTargetVectorSize() const;

    void setInputValue( const quint32 index, const uchar value );
    void setTargetValue( const quint32 index, const uchar value );
    uchar getInputValue( const quint32 index ) const;
    uchar getTargetValue( const quint32 index ) const;

    void setMark( const QString& mark );
    const QString& getMark() const;

    template< class Iterator >
    void setInputVector( Iterator first, Iterator last )
    {
        setInputVectorSize( last - first );
        std::copy( first, last, input );
    }

    template< class Iterator >
    void setTargetVector( Iterator first, Iterator last )
    {
        setTargetVectorSize( last - first );
        std::copy( first, last, target );
    }

private:
    uchar* input;
    uchar* target;
    quint32 inputSize;
    quint32 targetSize;
    QString mark;
};

#endif /// NEURAL_NETWORK_LEARNING_SAMPLE_HPP
