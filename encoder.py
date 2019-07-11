import tensorflow as tf
import numpy as np
from tensorflow.python.framework import ops
import json
from transformer_encoder import trans_encoder_config, transformer_encoder


def Bi_lstm_encoder(cell_fw, cell_bw, encoder_inputs_embedded):  
	(outputs,outputs_state) = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, encoder_inputs_embedded,dtype=tf.float32)  #time_major: False (default)
	(output_fw,output_bw) = outputs  
	outputs = tf.concat(outputs, 2) 
			
	(output_state_fw,output_state_bw) = outputs_state
	(fw_c_state,fw_h_state) = output_state_fw
	(bw_c_state,bw_h_state) = output_state_bw
	c_state = tf.concat([fw_c_state, bw_c_state], -1)  
	h_state = tf.concat([fw_h_state, bw_h_state], -1)  
	return h_state,c_state,outputs


def transoformer(inputs, mask, num_block, num_head, intermediate_hidden_size):
	encoder_inputs_tensor = inputs
	encoder_input_mask = mask
	trans_config = trans_encoder_config(hidden_size=encoder_inputs_tensor.get_shape()[-1].value,  
				 						num_hidden_layers=num_block,        
				 						num_attention_heads=num_head,      
				 						intermediate_size=intermediate_hidden_size,    
				 						hidden_act="gelu",	       
				 						hidden_dropout_prob=0.1,     
				 						attention_probs_dropout_prob=0.1,  
				 						max_position_embeddings=512,     
				 						initializer_range=0.02)

	transformer_model = transformer_encoder(config=trans_config,
											s_training=False,
											input_emb=encoder_inputs_tensor, 
											input_mask=encoder_input_mask)    

	encoder_seq_out = transformer_model.get_sequence_output() 
	return encoder_seq_out