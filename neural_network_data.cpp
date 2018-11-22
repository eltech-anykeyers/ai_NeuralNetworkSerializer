#include "neural_network_data.hpp"

NeuralNetworkData::NeuralNetworkData()
{
    learningData = QVector< NeuralNetworkLearningSample >();
    relationshipsWeighs = Q_NULLPTR;
    weightsMatrixSize = imageSize = QSize( 0, 0 );
}

NeuralNetworkData::NeuralNetworkData( const quint32 nNeurons,
                                      const QSize& imageSize )
    : NeuralNetworkData()
{
    setLayerParams( nNeurons, imageSize );
}

NeuralNetworkData::NeuralNetworkData( const NeuralNetworkData& other )
{
    learningData = other.learningData;
    weightsMatrixSize = other.weightsMatrixSize;
    imageSize = other.imageSize;

    auto arraySize = quint32( weightsMatrixSize.width() * weightsMatrixSize.height() );
    relationshipsWeighs = new double[ arraySize ];
    std::copy( other.relationshipsWeighs,
               other.relationshipsWeighs + arraySize,
               relationshipsWeighs );
}

NeuralNetworkData::NeuralNetworkData( NeuralNetworkData&& other )
{
    learningData = std::move( other.learningData );
    weightsMatrixSize = std::move( other.weightsMatrixSize );
    imageSize = std::move( other.imageSize );
    relationshipsWeighs = other.relationshipsWeighs;

    other.relationshipsWeighs = Q_NULLPTR;
    other.weightsMatrixSize = other.imageSize = QSize( 0, 0 );
    other.learningData = QVector< NeuralNetworkLearningSample >();
}

NeuralNetworkData::~NeuralNetworkData()
{
    clearLayer();
    learningData.clear();
}

void NeuralNetworkData::setLayerParams( const quint32 nNeurons,
                                        const QSize& imageSize )
{
    clearLayer();
    this->imageSize = imageSize;
    quint32 inputSize = quint32( imageSize.height() * imageSize.width() );
    weightsMatrixSize = QSize( qint32( nNeurons ), qint32( inputSize ) );
    relationshipsWeighs = new double[ nNeurons * inputSize ];
}

quint32 NeuralNetworkData::getNumberOfNeurons() const
{
    return quint32( weightsMatrixSize.width() );
}

quint32 NeuralNetworkData::getInputSize() const
{
    return quint32( weightsMatrixSize.height() );
}

bool NeuralNetworkData::isNull() const
{
    return learningData.empty() &&
           !relationshipsWeighs &&
           weightsMatrixSize.isEmpty();
}

void NeuralNetworkData::setRelationshipWeight(
        quint32 neuronNo, quint32 inputNo, double value )
{
    if( neuronNo >= quint32( weightsMatrixSize.width() ) ||
        inputNo >= quint32( weightsMatrixSize.height() ) ) return;

    auto index = neuronNo * quint32( weightsMatrixSize.height() ) + inputNo;
    relationshipsWeighs[ index ] = value;
}

double NeuralNetworkData::getRelationshipWeight( quint32 neuronNo, quint32 inputNo ) const
{
    if( neuronNo >= quint32( weightsMatrixSize.width() ) ||
        inputNo >= quint32( weightsMatrixSize.height() ) ) return 0.0;

    auto index = neuronNo * quint32( weightsMatrixSize.height() ) + inputNo;
    return relationshipsWeighs[ index ];
}

const QSize& NeuralNetworkData::getImageSize() const
{
    return imageSize;
}

void NeuralNetworkData::clearLayer()
{
    if( relationshipsWeighs )
    {
        delete[] relationshipsWeighs;
        relationshipsWeighs = Q_NULLPTR;
        weightsMatrixSize = QSize( 0, 0 );
    }
}

void NeuralNetworkData::setLearningData(
        const QVector< NeuralNetworkLearningSample >& data )
{
    learningData.clear();
    for( const auto& item : data )
    {
        addLearningData( item );
    }
}

void NeuralNetworkData::addLearningData(
        const NeuralNetworkLearningSample& sample )
{
    if( learningData.empty() ||
        ( learningData.front().getInputVectorSize() == sample.getInputVectorSize() &&
          learningData.front().getTargetVectorSize() == sample.getTargetVectorSize() ) )
    {
        learningData.append( sample );
    }
}

const QVector< NeuralNetworkLearningSample >& NeuralNetworkData::getLearningData() const
{
    return learningData;
}
