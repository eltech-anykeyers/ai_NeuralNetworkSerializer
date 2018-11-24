#include "neural_network_weights_matrix.hpp"

NeuralNetworkWeightsMatrix::NeuralNetworkWeightsMatrix()
{
    relationshipsWeights = QVector< qreal >();
    weightsMatrixSize = QSize( 0, 0 );
}

NeuralNetworkWeightsMatrix::NeuralNetworkWeightsMatrix( const QSize& matrixSize )
    : NeuralNetworkWeightsMatrix()
{
    if( matrixSize.isValid() ) setLayerParams( matrixSize );
}

NeuralNetworkWeightsMatrix::NeuralNetworkWeightsMatrix(
        const quint32 nNeurons, const quint32 inputSize )
    : NeuralNetworkWeightsMatrix( QSize( qint32( nNeurons ),
                                         qint32( inputSize ) ) )
{}

void NeuralNetworkWeightsMatrix::setLayerParams( const QSize& matrixSize )
{
    this->weightsMatrixSize = matrixSize;
    relationshipsWeights.resize(
        this->weightsMatrixSize.width() * this->weightsMatrixSize.height() );
}

quint32 NeuralNetworkWeightsMatrix::getMatrixWidth() const
{
    return quint32( weightsMatrixSize.width() );
}

quint32 NeuralNetworkWeightsMatrix::getMatrixHeight() const
{
    return quint32( weightsMatrixSize.height() );
}

bool NeuralNetworkWeightsMatrix::isNull() const
{
    return relationshipsWeights.isEmpty() &&
           weightsMatrixSize.isEmpty();
}

void NeuralNetworkWeightsMatrix::setRelationshipWeight(
        quint32 neuronNo, quint32 inputNo, qreal value )
{
    if( neuronNo >= quint32( weightsMatrixSize.width() ) ||
        inputNo >= quint32( weightsMatrixSize.height() ) ) return;

    auto index = qint32( neuronNo ) * weightsMatrixSize.height() + qint32( inputNo );
    relationshipsWeights[ index ] = value;
}

double NeuralNetworkWeightsMatrix::getRelationshipWeight( quint32 neuronNo, quint32 inputNo ) const
{
    if( neuronNo >= quint32( weightsMatrixSize.width() ) ||
        inputNo >= quint32( weightsMatrixSize.height() ) ) return 0.0;

    auto index = qint32( neuronNo ) * weightsMatrixSize.height() + qint32( inputNo );
    return relationshipsWeights[ index ];
}

void NeuralNetworkWeightsMatrix::clear()
{
    relationshipsWeights.clear();
    weightsMatrixSize = QSize( 0, 0 );
}
