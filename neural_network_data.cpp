#include "neural_network_data.hpp"

NeuralNetworkData::NeuralNetworkData()
{
    learningData = QVector< NeuralNetworkLearningSample >();
    layers = QVector< NeuralNetworkWeightsMatrix >();
}

NeuralNetworkData::NeuralNetworkData( const QSize& imageSize )
    : NeuralNetworkData()
{
    setImageSize( imageSize );
}

void NeuralNetworkData::setImageSize( const QSize& imageSize )
{
    this->imageSize = imageSize;
}

const QSize& NeuralNetworkData::getImageSize() const
{
    return imageSize;
}

bool NeuralNetworkData::isNull() const
{
    return learningData.empty() && layers.empty();
}

void NeuralNetworkData::clear()
{
    learningData.clear();
    layers.clear();
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

void NeuralNetworkData::setNeuralNetworkLayers( const QVector< NeuralNetworkWeightsMatrix >& layers )
{
    learningData.clear();
    for( const auto& item : layers )
    {
        addNeuralNetworkLayer( item );
    }
}

void NeuralNetworkData::addNeuralNetworkLayer( const NeuralNetworkWeightsMatrix& layer )
{
    layers.append( layer );
}

const QVector< NeuralNetworkLearningSample >& NeuralNetworkData::getLearningData() const
{
    return learningData;
}

const QVector< NeuralNetworkWeightsMatrix >& NeuralNetworkData::getNeuralNetworkLayers() const
{
    return layers;
}
