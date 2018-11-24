#ifndef NEURAL_NETWORK_DATA_HPP
#define NEURAL_NETWORK_DATA_HPP

#include <QVector>
#include <QSize>
#include <QString>

#include "neural_network_learning_sample.hpp"
#include "neural_network_weights_matrix.hpp"

class NeuralNetworkData
{
public:
    NeuralNetworkData();
    explicit NeuralNetworkData( const QSize& imageSize );
    ~NeuralNetworkData() = default;

    void setImageSize( const QSize& imageSize );
    const QSize& getImageSize() const;

    bool isNull() const;
    void clear();

    void setLearningData( const QVector< NeuralNetworkLearningSample >& data );
    void addLearningData( const NeuralNetworkLearningSample& sample );
    void setNeuralNetworkLayers( const QVector< NeuralNetworkWeightsMatrix >& layers );
    void addNeuralNetworkLayer( const NeuralNetworkWeightsMatrix& layer );

    const QVector< NeuralNetworkLearningSample >& getLearningData() const;
    const QVector< NeuralNetworkWeightsMatrix >& getNeuralNetworkLayers() const;

private:
    QVector< NeuralNetworkLearningSample > learningData;
    QVector< NeuralNetworkWeightsMatrix > layers;
    QSize imageSize;
};

#endif /// NEURAL_NETWORK_DATA_HPP
