#include "neural_network_data.hpp"

NeuralNetworkData::NeuralNetworkData()
{
    learningData = QVector< NeuralNetworkLearningSample >();
    layers = QVector< NeuralNetworkWeightsMatrix >();
    meta = QByteArray();
}

bool NeuralNetworkData::isNull() const
{
    return learningData.empty() && layers.empty() &&
            meta.size() == 0;
}

void NeuralNetworkData::clear()
{
    learningData.clear();
    layers.clear();
    meta.clear();
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

void NeuralNetworkData::setMetaInformation( const QByteArray& metaInfo )
{
    meta = metaInfo;
}

const QVector< NeuralNetworkLearningSample >& NeuralNetworkData::getLearningData() const
{
    return learningData;
}

const QVector< NeuralNetworkWeightsMatrix >& NeuralNetworkData::getNeuralNetworkLayers() const
{
    return layers;
}

const QByteArray& NeuralNetworkData::getMetaInformation() const
{
    return meta;
}
