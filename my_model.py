import tensorflow as tf
import numpy as np
from tensorflow.python.framework import ops
import json
from encoder import Bi_lstm_encoder,transoformer
print(tf.__version__)
print(tf.__path__)

#self_Normalization——selu
def selu(x):   #input one data
	with ops.name_scope('selu') as scope:
		alpha = 1.6732632423543772848170429916717
		scale = 1.0507009873554804934193349852946
		return scale * tf.where(x >= 0.0, x, alpha * tf.nn.elu(x))

#load pretrain wordembedding
def get_pre_word_emb(filename):
	print('Word Embedding init from %s' % filename)
	words_id2vec = json.load(open(filename, 'r'))
	words_vectors = [] 
	for id, vec in words_id2vec.items():
		vec = np.array(vec)
		words_vectors.append(vec)
	words_vectors = np.array(words_vectors)
	words_embedding_table = tf.Variable(name='words_emb_table', initial_value=words_vectors, dtype=tf.float32)
	return words_embedding_table

class Seq2SeqModel():
	def __init__(self,pre_word_embedding, vocab_to_idx, relation_to_idx,
						max_source_length,max_target_length,batch_size,
						rnn_size, num_layers, embedding_size,gated_hidden_size,
						mode, use_attention):
		self.pre_word_embedding = pre_word_embedding
		self.vocab_to_idx = vocab_to_idx
		self.relation_to_idx = relation_to_idx
		self.vocab_size = len(self.vocab_to_idx)
		self.realtion_size = len(self.relation_to_idx)

		self.max_source_length = max_source_length
		self.max_target_length = max_target_length
		self.batch_size = batch_size
		
		self.mode = mode   #train/decoder
		self.embedding_size = embedding_size
		self.rnn_size = rnn_size
		self.num_layers = num_layers
		self.gated_hidden_size = gated_hidden_size

		self.use_attention = use_attention   #true/false
		
		self.build_model()
	
	#encoder_cell
	def _create_rnn_cell(self):
		single_cell = tf.contrib.rnn.LSTMCell(self.rnn_size)
		return single_cell

	#decoder_cell
	def create_new_cell(self): 
		one_cell = tf.contrib.rnn.LSTMCell(2*self.rnn_size,state_is_tuple=True)
		return one_cell


	#Bahdanau Attention
	def attention_c(self,hidden_state,gated_outputs,input_mask):

		tile_num = int(gated_outputs.get_shape()[1])   
		hidden_state_shape_0 = int(hidden_state.get_shape()[0]) 
		hidden_state_shape_1 = int(hidden_state.get_shape()[1]) 
		with tf.variable_scope('attention'):

			h_s = tf.reshape(tf.tile(hidden_state,[1,tile_num]),[hidden_state_shape_0,-1,hidden_state_shape_1])
			W_att = tf.get_variable(name='W_att',shape=[hidden_state_shape_1,hidden_state_shape_1],dtype=tf.float32)
			W_att = tf.reshape(W_att,[-1,1])
			W_att = tf.transpose(tf.tile(W_att,[1,hidden_state_shape_0]))
			W_att = tf.reshape(W_att,[hidden_state_shape_0,-1,hidden_state_shape_1])
			# h_s * W_att
			mat = tf.matmul(h_s,W_att)  #[batch_size, max_time, 2*cell_size]

			dot_operation = tf.multiply(mat, gated_outputs)  #  [batch_size, max_time, 2*cell_size]  
			dot_sum = tf.reduce_sum(dot_operation, axis=-1)  # [batch_size, max_time]    
			adder = (1.0 - tf.cast(input_mask, tf.float32)) * -10000.0
			dot_sum = dot_sum + adder

			#attention 
			atten = tf.nn.softmax(dot_sum) 

			att_out =tf.multiply(tf.expand_dims(atten,-1),gated_outputs)   
			c = tf.reduce_sum(att_out, axis=1) #[batch_size, 2*cell_size]
			return c,atten


	# head_pointer
	def head_ptr_logits(self,inputs):
		with tf.variable_scope('head_ptr_logits',reuse=tf.AUTO_REUSE):
			self.W_h = tf.get_variable(name='W_h',shape=[int(inputs.get_shape()[-1]), self.max_source_length+1], dtype=tf.float32)# max_inputs_length + <eos>
			self.b_h = tf.get_variable(name='b_h',shape=self.max_source_length+1,dtype=tf.float32)
			logits_hptr = selu(tf.matmul(inputs, self.W_h) + self.b_h)
			return logits_hptr     #[B,max_input_length+1]  

	# tail_pointer
	def tail_ptr_logits(self,inputs):
		with tf.variable_scope('tail_ptr_logits',reuse=tf.AUTO_REUSE):
			self.W_t = tf.get_variable(name='W_t',shape=[int(inputs.get_shape()[-1]), 4],dtype=tf.float32)      
			self.b_t = tf.get_variable(name='b_t',shape=4, dtype=tf.float32)
			logits_tptr = selu(tf.matmul(inputs, self.W_t) + self.b_t)
			return logits_tptr 
	#relation
	def relation_logits(self,inputs):
		with tf.variable_scope('relation_logits',reuse=tf.AUTO_REUSE):
			self.W_r = tf.get_variable(name='W_r',shape=[int(inputs.get_shape()[-1]), self.realtion_size],dtype=tf.float32)
			self.b_r = tf.get_variable(name='b_r',shape=self.realtion_size,dtype=tf.float32)
			logits_r = selu(tf.matmul(inputs, self.W_r) + self.b_r)
			return logits_r

	#probility
	def pred_prob(self,inputs,targets):
		depth = inputs.get_shape()[-1]   
		indexes = tf.reshape(targets,[-1])  
		one_hot = tf.one_hot(indexes, depth) 
		probs = tf.reduce_sum(inputs * one_hot, axis=1) 
		return probs


	def build_model(self):
		print('building model... ...')

		self.encoder_inputs = tf.placeholder(tf.int32, [self.batch_size, self.max_source_length+1], name='encoder_inputs')
		self.encoder_inputs_length = tf.placeholder(tf.int32, [self.batch_size], name='encoder_inputs_length')

		self.decoder_targets = tf.placeholder(tf.int32, [self.batch_size, self.max_target_length], name='decoder_targets')
		self.decoder_targets_length = tf.placeholder(tf.int32, [self.batch_size], name='decoder_targets_length')  

		self.keep_prob_placeholder = tf.placeholder(tf.float32, name='keep_prob_placeholder')
		self.learing_rate = tf.placeholder(tf.float32, name='learing_rate')
	
		#input_mask
		self.input_mask = tf.sequence_mask(self.encoder_inputs_length, self.max_source_length + 1, dtype=tf.float32, name='mask1')

		self.target_mask = tf.sequence_mask(self.decoder_targets_length, self.max_target_length, dtype=tf.float32, name='mask2')

		vocab_embedding = tf.get_variable('vocab_embedding', [self.vocab_size, self.embedding_size])
		relation_embedding = tf.get_variable('relation_embedding', [self.realtion_size, self.embedding_size])

		encoder_inputs_embedded = tf.nn.embedding_lookup(self.pre_word_embedding, self.encoder_inputs)

		with tf.variable_scope('encoder'): 

			encoder_input_mask = tf.sequence_mask(self.encoder_inputs_length, self.max_source_length+1, dtype=tf.int32)


			position_embedding = tf.get_variable('position_embedding', [self.max_source_length+1, 100])
			in_batch = int(encoder_inputs_embedded.get_shape()[0])
			in_hidden = int(position_embedding.get_shape()[-1])
			in_pos = tf.reshape(position_embedding,[-1,1])
			in_pos = tf.transpose(tf.tile(in_pos,[1,in_batch]))
			in_pos = tf.reshape(in_pos,[in_batch,-1,in_hidden]) 
			encoder_inputs_embedded = tf.concat([encoder_inputs_embedded,in_pos],-1)
			

			outputs = transoformer(encoder_inputs_embedded, encoder_input_mask,num_block=4, num_head=4, intermediate_hidden_size=512)
			h_state = tf.zeros([outputs.get_shape()[0], outputs.get_shape()[-1]],dtype = tf.float32)
			c_state = tf.zeros([outputs.get_shape()[0], outputs.get_shape()[-1]],dtype = tf.float32)
  
		
		with tf.variable_scope('decoder'):
			self.pointer_cell = self.create_new_cell()  
			self.relation_cell = self.create_new_cell()  

			self.prob_by_time = []   
			self.select_by_time = [] 
			self.attention_by_time = []
	
			with tf.variable_scope('rnn'):
				
				go_idx = 0  #<go>
				cut_end = tf.strided_slice(self.decoder_targets, [0, 0], [self.batch_size, -1], [1, 1])  
				decoder_inputs = tf.concat([tf.fill([self.batch_size, 1], go_idx), cut_end], 1)  #<go>

				decoder_targets = self.decoder_targets
				decoder_state = (c_state,h_state)

				if self.mode == 'test':
					decoder_step_inputs = tf.fill([self.batch_size, 1], go_idx)  
					decoder_step_inputs = tf.cast(decoder_step_inputs, tf.int32)

				for time_step in range(self.max_target_length):
					if time_step > 0:
						tf.get_variable_scope().reuse_variables()

					#attention 
					gated_c,atten_step = self.attention_c(decoder_state[1],outputs,self.input_mask)
					
					if time_step % 5 == 0:  #relation
						decoder_inputs_embedded = tf.nn.embedding_lookup(relation_embedding, decoder_inputs[:,time_step])
					if time_step % 5 == 1 or time_step % 5 == 3:#head_pointer
						index = tf.expand_dims(decoder_inputs[:,time_step],-1)  
						index = tf.stack([tf.range(index.shape[0])[:, None], index], axis=2)
						index = tf.clip_by_value(index, 0, self.max_source_length)  
						result = tf.gather_nd(outputs, index)   
						decoder_inputs_embedded = tf.identity(result)
						decoder_inputs_embedded = tf.reduce_sum(decoder_inputs_embedded,axis=1)

					if time_step % 5 == 2 or time_step % 5 == 4:#tail_pointer
						h_index = decoder_inputs[:,time_step-1]
						index = decoder_inputs[:,time_step]
						index = h_index + index
						index = tf.clip_by_value(index, 0, self.max_source_length)   
						index = tf.expand_dims(index,-1)  
						index = tf.stack([tf.range(index.shape[0])[:, None], index], axis=2)
						result = tf.gather_nd(outputs, index)   
						decoder_inputs_embedded = tf.identity(result)
						decoder_inputs_embedded = tf.reduce_sum(decoder_inputs_embedded,axis=1)

					decoder_cell_inputs = decoder_inputs_embedded


					if self.mode == 'test':
						if time_step % 5 == 0:  #relation
							decoder_inputs_embedded = tf.nn.embedding_lookup(relation_embedding, decoder_step_inputs[:,time_step])

						if time_step % 5 == 1 or time_step % 5 == 3:#head_pointer
							index = tf.expand_dims(decoder_step_inputs[:,time_step],-1)   
							index = tf.stack([tf.range(index.shape[0])[:, None], index], axis=2)
							index = tf.clip_by_value(index, 0, self.max_source_length)   
							result = tf.gather_nd(outputs, index)   
							decoder_inputs_embedded = tf.identity(result)
							decoder_inputs_embedded = tf.reduce_sum(decoder_inputs_embedded,axis=1)

						if time_step % 5 == 2 or time_step % 5 == 4:#tail_pointer
							h_index = decoder_step_inputs[:,time_step-1]
							index = decoder_step_inputs[:,time_step]
							index = h_index + index        
							index = tf.clip_by_value(index, 0, self.max_source_length)   
							index = tf.expand_dims(index,-1)  
							index = tf.stack([tf.range(index.shape[0])[:, None], index], axis=2)
							result = tf.gather_nd(outputs, index)  
							decoder_inputs_embedded = tf.identity(result)
							decoder_inputs_embedded = tf.reduce_sum(decoder_inputs_embedded,axis=1)
						decoder_cell_inputs = decoder_inputs_embedded
						
					#concat input & c
					cell_inputs = tf.concat([decoder_cell_inputs,gated_c],-1)   #[batch_size,embedding_size+2*cell_size] 

					#call cell
					if time_step % 5 != 0:  #pointer
						with tf.variable_scope('pointer_decoder_cell',reuse=tf.AUTO_REUSE):
							(decoder_output, decoder_state) = self.pointer_cell(cell_inputs, decoder_state) 
					elif time_step % 5 == 0:  #relation
						with tf.variable_scope('relation_decoder_cell',reuse=tf.AUTO_REUSE): 
							(decoder_output, decoder_state) = self.relation_cell(cell_inputs, decoder_state) 

					#classfication_logits
					# head_pointer
					if time_step % 5 == 0 or time_step % 5 == 2:
						step_logits = self.head_ptr_logits(decoder_output)  
						step_prob = step_logits* self.input_mask          
						step_prob = tf.nn.softmax(step_prob)
						
					# tail_pointer
					if time_step % 5 == 1 or time_step % 5 == 3:
						step_logits = self.tail_ptr_logits(decoder_output)  
						step_prob = tf.nn.softmax(step_logits)

					# relation
					if time_step % 5 == 4:
						step_logits = self.relation_logits(decoder_output) 
						step_prob = tf.nn.softmax(step_logits)

					select_index = tf.argmax(step_prob, axis=1)  
					select_index = tf.cast(select_index, tf.int32)

					pred_prob = self.pred_prob(step_prob,decoder_targets[:,time_step])  

					if self.mode == 'test':
						select_index = tf.reshape(select_index,[-1,1])
						select_index = tf.cast(select_index, tf.int32)
						decoder_step_inputs = tf.concat([decoder_step_inputs,select_index],-1)
						
					self.prob_by_time.append(pred_prob)
					self.select_by_time.append(select_index)
					self.attention_by_time.append(atten_step)

				self.select_by_time = tf.stack(self.select_by_time, axis=1)	
				self.attention_by_time = tf.stack(self.attention_by_time, axis=1)	
				if self.mode == 'test':
					self.select_by_time  = tf.reshape(self.select_by_time,[int(self.select_by_time.shape[0]),-1])


				self.prob_by_time = tf.stack(self.prob_by_time, axis=1)	

				prob = tf.identity(self.prob_by_time)  
				prob = tf.clip_by_value(prob, 1e-8, 1.0)    			

				with tf.name_scope('loss'):
					targets_len = tf.cast(tf.reduce_sum(self.decoder_targets_length), tf.float32)
					log_prob = -tf.log(prob)   
					log_prob = log_prob * self.target_mask   
					log_prob = tf.reshape(log_prob, [-1])   

					self.losses = tf.reduce_sum(log_prob)/targets_len
					self.loss_scalar = tf.summary.scalar('loss', self.losses)

			self.optimizer = tf.train.AdamOptimizer(self.learing_rate)
			self.train_op = self.optimizer.minimize(self.losses)
			self.saver = tf.train.Saver(tf.global_variables())			


	def train(self, sess, batch, lr):
		feed_dict = {self.encoder_inputs: np.array(batch.encoder_inputs),
						self.encoder_inputs_length: np.array(batch.encoder_inputs_length),
						self.decoder_targets: np.array(batch.decoder_targets),
						self.decoder_targets_length: np.array(batch.decoder_targets_length),
						self.keep_prob_placeholder: 0.9,
						self.learing_rate:lr}
		_,loss,select_by_time= sess.run([self.train_op,self.losses,self.select_by_time], feed_dict=feed_dict)
		loss_metall = sess.run(self.loss_scalar, feed_dict=feed_dict)   
		return loss,select_by_time,loss_metall

	def test(self, sess, batch,lr):
		feed_dict2 = {self.encoder_inputs: np.array(batch.encoder_inputs),
						self.encoder_inputs_length: np.array(batch.encoder_inputs_length),
						self.decoder_targets: np.array(batch.decoder_targets),
						self.decoder_targets_length: np.array(batch.decoder_targets_length),
						self.keep_prob_placeholder: 1.0,
						self.learing_rate:lr}
		loss,select_by_time,attention_by_time= sess.run([self.losses,self.select_by_time,self.attention_by_time], feed_dict=feed_dict2)
		loss_metall = sess.run(self.loss_scalar, feed_dict=feed_dict2)
		return loss,select_by_time,attention_by_time,loss_metall




					


					
					

				
					










 


