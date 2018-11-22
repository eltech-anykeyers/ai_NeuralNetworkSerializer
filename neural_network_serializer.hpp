#ifndef NEURAL_NETWORK_SERIALIZER_HPP
#define NEURAL_NETWORK_SERIALIZER_HPP

#include "neural_network_data.hpp"

#include <QObject>
#include <QIODevice>

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

private:
    std::unique_ptr< QIODevice > device;
};

#endif /// NEURAL_NETWORK_SERIALIZER_HPP
