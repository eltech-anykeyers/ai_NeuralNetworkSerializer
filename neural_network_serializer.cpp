#include "neural_network_serializer.hpp"
#include "neural_network_learning_sample.hpp"

#include <QDataStream>

NeuralNetworkSerializer::NeuralNetworkSerializer( QObject* parent )
    : QObject( parent )
{
    device = Q_NULLPTR;
}

void NeuralNetworkSerializer::serialize( const NeuralNetworkData& data ) const
{
    if( !device ) return;

    device->open( QIODevice::WriteOnly );
    QDataStream stream = QDataStream( device.get() );

    /// Size of weights matrix.
    stream << static_cast< quint32 >( data.getNumberOfNeurons() );
    stream << static_cast< quint32 >( data.getInputSize() );

    /// Image size.
    stream << static_cast< quint32 >( data.getImageSize().width() );
    stream << static_cast< quint32 >( data.getImageSize().height() );

    /// Weights matrix.
    for( quint32 i = 0; i < data.getNumberOfNeurons(); i++ )
    {
        for( quint32 j = 0; j < data.getInputSize(); j++ )
        {
            stream << data.getRelationshipWeight( i, j );
        }
    }

    /// Number of samples.
    stream << static_cast< quint32 >( data.getLearningData().size() );

    stream << data.getLearningData().front().getTargetVectorSize();
    stream << data.getLearningData().front().getInputVectorSize();

    /// Samples.
    for( const auto& sample : data.getLearningData() )
    {
        /// Target vector.
        for( quint32 i = 0; i < sample.getTargetVectorSize(); i++ )
        {
            stream << sample.getTargetValue( i );
        }

        /// Input vector.
        for( quint32 j = 0; j < sample.getInputVectorSize(); j++ )
        {
            stream << sample.getInputValue( j );
        }

        /// Mark.
        stream << sample.getMark().toStdString().c_str();
    }

    device->close();
}

NeuralNetworkData NeuralNetworkSerializer::deserialize() const
{
    if( !device ) return NeuralNetworkData();

    device->open( QIODevice::ReadOnly );
    QDataStream stream = QDataStream( device.get() );

    NeuralNetworkData result;

    /// Size of weights matrix.
    quint32 nNeurons, inputSize;
    stream >> nNeurons;
    stream >> inputSize;

    /// Image size.
    quint32 imageWidth, imageHeight;
    stream >> imageWidth;
    stream >> imageHeight;
    if( imageHeight * imageWidth != inputSize )
    {
        return NeuralNetworkData();
    }
    result.setLayerParams(  nNeurons,
                            QSize( qint32( imageWidth ), qint32( imageHeight ) ) );

    /// Weights matrix.
    double weight;
    for( quint32 i = 0; i < result.getNumberOfNeurons(); i++ )
    {
        for( quint32 j = 0; j < result.getInputSize(); j++ )
        {
            stream >> weight;
            result.setRelationshipWeight( i, j, weight );
        }
    }

    /// Number of samples.
    quint32 nSamples;
    stream >> nSamples;

    /// Sample sizes.
    quint32 sampleTargetSize, sampleInputSize;
    stream >> sampleTargetSize;
    stream >> sampleInputSize;

    /// Samples.
    for( int k = 0; k < static_cast< qint32 >( nSamples ); k++ )
    {
        /// Create sample.
        NeuralNetworkLearningSample sample;
        sample.setInputVectorSize( sampleInputSize );
        sample.setTargetVectorSize( sampleTargetSize );

        /// Target vector.
        uchar value;
        for( quint32 i = 0; i < sample.getTargetVectorSize(); i++ )
        {
            stream >> value;
            sample.setTargetValue( i, value );
        }

        /// Input vector.
        for( quint32 i = 0; i < sample.getInputVectorSize(); i++ )
        {
            stream >> value;
            sample.setInputValue( i, value );
        }

        /// Mark.
        char* mark;
        stream >> mark;
        sample.setMark( QString( mark ) );
        delete[] mark;

        result.addLearningData( std::move( sample ) );
    }

    device->close();
    return result;
}

void NeuralNetworkSerializer::setDevice( std::unique_ptr< QIODevice > device )
{
    this->device = std::move( device );
}

const QIODevice& NeuralNetworkSerializer::getDevice() const
{
    return *device;
}
