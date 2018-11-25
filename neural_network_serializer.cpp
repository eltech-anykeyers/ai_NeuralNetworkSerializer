#include "neural_network_serializer.hpp"
#include "neural_network_learning_sample.hpp"

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

    writeHeader( stream );
    writeNeuralNetworkLayers( stream, data.getNeuralNetworkLayers() );
    writeNeuralNetworkLearningSamples( stream, data.getLearningData() );
    writeNeuralNetworkMetaInformation( stream, data.getMetaInformation() );

    device->close();
}

NeuralNetworkData NeuralNetworkSerializer::deserialize() const
{
    if( !device ) return NeuralNetworkData();

    device->open( QIODevice::ReadOnly );
    QDataStream stream = QDataStream( device.get() );

    auto verified = verifyHeader( stream );
    if( !verified ) return NeuralNetworkData();

    NeuralNetworkData result;

    readNeuralNetworkLayers( stream, result );
    readNeuralNetworkLearningSamples( stream, result );
    readNeuralNetworkMetaInformation( stream, result );

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

void NeuralNetworkSerializer::writeHeader( QDataStream& stream ) const
{
    ///TODO
}

void NeuralNetworkSerializer::writeNeuralNetworkLayers(
        QDataStream& stream, const QVector< NeuralNetworkWeightsMatrix >& layers ) const
{
    /// Number of layers.
    stream << static_cast< quint32 >( layers.size() );

    /// Layers.
    for( const auto& layer : layers )
    {
        /// Size of weights matrix.
        stream << static_cast< quint32 >( layer.getMatrixWidth() );
        stream << static_cast< quint32 >( layer.getMatrixHeight() );

        /// Weights matrix.
        for( quint32 i = 0; i < layer.getMatrixWidth(); ++i )
        {
            for( quint32 j = 0; j < layer.getMatrixHeight(); ++j )
            {
                stream << layer.getRelationshipWeight( i, j );
            }
        }
    }
}

void NeuralNetworkSerializer::writeNeuralNetworkLearningSamples(
        QDataStream& stream, const QVector< NeuralNetworkLearningSample >& samples ) const
{
    /// Number of samples.
    stream << static_cast< quint32 >( samples.size() );

    /// Samples params.
    stream << samples.front().getTargetVectorSize();
    stream << samples.front().getInputVectorSize();

    /// Samples.
    for( const auto& sample : samples )
    {
        /// Target vector.
        for( quint32 i = 0; i < sample.getTargetVectorSize(); ++i )
        {
            stream << sample.getTargetValue( i );
        }

        /// Input vector.
        for( quint32 j = 0; j < sample.getInputVectorSize(); ++j )
        {
            stream << sample.getInputValue( j );
        }

        /// Mark.
        stream << sample.getMark().toStdString().c_str();
    }
}

void NeuralNetworkSerializer::writeNeuralNetworkMetaInformation(
        QDataStream& stream, const QByteArray& metaInfo ) const
{
    stream << static_cast< quint32 >( metaInfo.size() );
    for( const auto& byte : metaInfo )
    {
        stream << static_cast< uchar >( byte );
    }
}

bool NeuralNetworkSerializer::verifyHeader( QDataStream& stream ) const
{
    ///TODO
    return true;
}

void NeuralNetworkSerializer::readNeuralNetworkLayers(
        QDataStream& stream, NeuralNetworkData& data ) const
{
    /// Number of layers.
    quint32 nLayers;
    stream >> nLayers;

    /// Layers.
    for( int k = 0; k < static_cast< qint32 >( nLayers ); ++k )
    {
        /// Size of weights matrix.
        quint32 nNeurons, inputSize;
        stream >> nNeurons;
        stream >> inputSize;

        /// Create layer.
        NeuralNetworkWeightsMatrix layer( nNeurons, inputSize );

        /// Weights matrix.
        double weight;
        for( quint32 i = 0; i < nNeurons; ++i )
        {
            for( quint32 j = 0; j < inputSize; ++j )
            {
                stream >> weight;
                layer.setRelationshipWeight( i, j, weight );
            }
        }

        data.addNeuralNetworkLayer( layer );
    }
}

void NeuralNetworkSerializer::readNeuralNetworkLearningSamples(
        QDataStream& stream, NeuralNetworkData& data  ) const
{
    /// Number of samples.
    quint32 nSamples;
    stream >> nSamples;

    /// Sample params.
    quint32 sampleTargetSize, sampleInputSize;
    stream >> sampleTargetSize;
    stream >> sampleInputSize;

    /// Samples.
    for( int k = 0; k < static_cast< qint32 >( nSamples ); ++k )
    {
        /// Create sample.
        NeuralNetworkLearningSample sample;
        sample.setInputVectorSize( sampleInputSize );
        sample.setTargetVectorSize( sampleTargetSize );

        /// Target vector.
        uchar value;
        for( quint32 i = 0; i < sample.getTargetVectorSize(); ++i )
        {
            stream >> value;
            sample.setTargetValue( i, value );
        }

        /// Input vector.
        for( quint32 i = 0; i < sample.getInputVectorSize(); ++i )
        {
            stream >> value;
            sample.setInputValue( i, value );
        }

        /// Mark.
        char* mark;
        stream >> mark;
        sample.setMark( QString( mark ) );
        delete[] mark;

        data.addLearningData( std::move( sample ) );
    }
}

void NeuralNetworkSerializer::readNeuralNetworkMetaInformation(
        QDataStream& stream, NeuralNetworkData& data ) const
{
    /// Number of bytes.
    quint32 nBytes;
    stream >> nBytes;

    QByteArray metaInfo;
    metaInfo.resize( qint32( nBytes ) );

    /// Bytes.
    uchar byte;
    for( quint32 i = 0; i < nBytes; ++i )
    {
        stream >> byte;
        metaInfo[ i ] = static_cast< char >( byte );
    }

    data.setMetaInformation( metaInfo );
}
