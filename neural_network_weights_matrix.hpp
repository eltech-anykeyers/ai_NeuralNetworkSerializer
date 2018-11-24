#ifndef NEURAL_NETWORK_WEIGHTS_MATRIX_HPP
#define NEURAL_NETWORK_WEIGHTS_MATRIX_HPP

#include <QVector>
#include <QSize>

class NeuralNetworkWeightsMatrix
{
public:
    NeuralNetworkWeightsMatrix();
    explicit NeuralNetworkWeightsMatrix( const QSize& matrixSize );
    explicit NeuralNetworkWeightsMatrix(
            const quint32 nNeurons, const quint32 inputSize );
    ~NeuralNetworkWeightsMatrix() = default;

    void setLayerParams( const QSize& matrixSize );
    quint32 getMatrixWidth() const;
    quint32 getMatrixHeight() const;
    bool isNull() const;
    void clear();

    void setRelationshipWeight( quint32 neuronNo, quint32 inputNo,
                                qreal value );
    double getRelationshipWeight( quint32 neuronNo, quint32 inputNo ) const;

private:
    QVector< qreal > relationshipsWeights;
    QSize weightsMatrixSize;
};

#endif /// NEURAL_NETWORK_WEIGHTS_MATRIX_HPP
