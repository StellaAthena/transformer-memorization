import tensorflow as tf

class TFrecordCreator:
    """
    Creates TFRecord Dataset to store the memorization metric
    """
    def __init__(self,path):
        self.path = path
        self.writer = tf.io.TFRecordWriter(path)
    
    def _int64_feature(self,value):
        """Returns an int64_list from a bool / enum / int / uint."""
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
    
    def _serialize_example(self,value):
        """
        Creates a tf.train.Example message ready to be written to a file.
        """
        feature = {
            'result':self._int64_feature(value)
        }
        
        example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
        return example_proto.SerializeToString()
    
    def write(self,value):
        value = self._serialize_example(value)
        self.writer.write(value)
    
    def close(self):
        """
        Closes the tfrecord stream
        For some reason, using __del__ doesn't work
        """
        self.writer.close()




class TFRecordLoader:
    def __init__(self,path):
        self.path = path
        self.reader = tf.data.TFRecordDataset([path])
        self.reader = self.reader.map(self._parse_fn)
    
    def _parse_fn(self,example_proto):

        feature_description = {
            'result':tf.io.FixedLenFeature([], tf.int64, default_value=0)
        }
        return tf.io.parse_single_example(example_proto, feature_description)['result']
    def __iter__(self):
        return iter(self.reader)

if __name__ == '__main__':
    rec = TFrecordCreator('temp.tfrecord')
    for i in range(10):
        rec.write(i)
    
    rec.close()
    reader = TFRecordLoader('temp.tfrecords')
    for i in reader:
        print(i.numpy(),end=" ")