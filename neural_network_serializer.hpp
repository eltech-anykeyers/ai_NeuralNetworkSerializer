#ifndef NEURAL_NETWORK_SERIALIZER_HPP
#define NEURAL_NETWORK_SERIALIZER_HPP

#include "neural_network_data.hpp"

#include <QObject>
#include <QIODevice>
#include <QDataStream>

#include <memory>

class NeuralNetworkSerializer : public QObject
{
    Q_OBJECT
public:
    explicit NeuralNetworkSerializer( QObject* parent = Q_NULLPTR );
    virtual ~NeuralNetworkSerializer() override = default;

    void serialize( const NeuralNetworkData& data ) const;
    NeuralNetworkData deserialize() const;
    void setDevice( std::unique_ptr< QIODevice > device );
    const QIODevice& getDevice() const;
    
signals:

public slots:

protected:
    void writeHeader( QDataStream& stream ) const;
    void writeNeuralNetworkLayers(
            QDataStream& stream, const QVector< NeuralNetworkWeightsMatrix >& layers ) const;
    void writeNeuralNetworkLearningSamples(
            QDataStream& stream, const QVector< NeuralNetworkLearningSample >& samples ) const;
    void writeNeuralNetworkMetaInformation(
            QDataStream& stream, const QByteArray& metaInfo  ) const;
    bool verifyHeader( QDataStream& stream ) const;
    void readNeuralNetworkLayers(
            QDataStream& stream, NeuralNetworkData& data ) const;
    void readNeuralNetworkLearningSamples(
            QDataStream& stream, NeuralNetworkData& data  ) const;
    void readNeuralNetworkMetaInformation(
            QDataStream& stream, NeuralNetworkData& data  ) const;

private:
    std::unique_ptr< QIODevice > device;
};

#endif /// NEURAL_NETWORK_SERIALIZER_HPP
