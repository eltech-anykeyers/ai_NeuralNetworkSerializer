#ifndef NEURAL_NETWORK_DATA_HPP
#define NEURAL_NETWORK_DATA_HPP

#include <QVector>
#include <QSize>
#include <QString>

#include "neural_network_learning_sample.hpp"

struct NeuralNetworkData
{
public:
    NeuralNetworkData();
    explicit NeuralNetworkData( const quint32 nNeurons, const QSize& imageSize );
    NeuralNetworkData( const NeuralNetworkData& other );
    NeuralNetworkData( NeuralNetworkData&& other );
    ~NeuralNetworkData();

    void setLayerParams( const quint32 nNeurons,
                         const QSize& imageSize );
    quint32 getNumberOfNeurons() const;
    quint32 getInputSize() const;
    bool isNull() const;

    void setRelationshipWeight( quint32 neuronNo, quint32 inputNo,
                                double value );
    double getRelationshipWeight( quint32 neuronNo, quint32 inputNo ) const;

    template< class Iterator >
    void setRelationshipsWeights( Iterator first, Iterator last,
                                  const quint32 nNeurons, const QSize& imageSize  )
    {
        if( last - first == nNeurons * imageSize.width() * imageSize.height() )
        {
            setLayerParams( nNeurons, imageSize );
            std::copy( first, last, relationshipsWeighs );
        }
    }

    const QSize& getImageSize() const;

    void clearLayer();

    void setLearningData( const QVector< NeuralNetworkLearningSample >& data );
    void addLearningData( const NeuralNetworkLearningSample& sample );

    const QVector< NeuralNetworkLearningSample >& getLearningData() const;

private:
    QVector< NeuralNetworkLearningSample > learningData;
    double* relationshipsWeighs;
    QSize weightsMatrixSize;
    QSize imageSize;
};

#endif /// NEURAL_NETWORK_DATA_HPP
