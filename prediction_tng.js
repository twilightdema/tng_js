var stem = require('stem-porter');

var process = function(model, sentences, languages, alphaValue, randomSeed) {

  var W = model.hypers.W;
  var T = model.hypers.T;
  var vocab = model.hypers.vocab;
  var vocabOrig = model.hypers.vocabOrig;

  var _alpha = model.priors.alpha;
  var beta = model.priors.beta;
  var gamma = model.priors.gamma;
  var delta = model.priors.delta;

  var theta = model.posteriors.theta;
  var phi = model.posteriors.phi;
  var psi = model.posteriors.psi;
  var sigma = model.posteriors.sigma;
  
  var n_zw = model.counters.n_zw;
  var m_zwv = model.counters.m_zwv;
  var p_zwk = model.counters.p_zwk;
  var n_z = model.counters.n_z;
  var m_zw = model.counters.m_zw;
  
  // Result is perplexity of the model
  var result = 0;

  // Index-encoded array of sentences, with each row containing the indices of the words in the vocabulary.
  var documents = new Array();
  // Hash of vocabulary words and the count of how many times each word has been seen.
  var f = {};
  // Vocabulary of unique words in their original form.
  for(var i=0;i<vocab.length;i++) {
    f[vocab[i]] = 1;
  }
  // Array of stop words
  languages = languages || Array('en'); 
  if (sentences && sentences.length > 0) {
    var stopwords = new Array();

    languages.forEach(function(value) {
        var stopwordsLang = require('./stopwords_' + value + ".js");
        stopwords = stopwords.concat(stopwordsLang.stop_words);
    });

    for(var i=0;i<sentences.length;i++) {
      if (sentences[i]=="") continue;
      documents[i] = new Array();

      var words = sentences[i] ? sentences[i].split(/[\s,\"]+/) : null;
      console.log('words = ' +JSON.stringify(words));
      
      if(!words) continue;
      for(var wc=0;wc<words.length;wc++) {
        var w=words[wc].toLowerCase();
        if(languages.indexOf('en') != -1)
          w=w.replace(/[^a-z\'A-Z0-9\u00C0-\u00ff ]+/g, '');
        var wStemmed = stem(w);
        if (w=="" || !wStemmed || w.length==1 || stopwords.indexOf(w.replace("'", "")) > -1 || stopwords.indexOf(wStemmed) > -1 || w.indexOf("http")==0) continue;
        if (f[wStemmed]) { 
            f[wStemmed]=f[wStemmed]+1;
            documents[i].push(vocab.indexOf(wStemmed));
          } 
        else if(wStemmed) { 
          // We use -1 to indicate verbatim that is not existing in our model dictionary.
          // documents[i].push(-1);
        };            
      }
    }

    var D = documents.length;
    var alpha = alphaValue || _alpha;  // per-document distributions over topics
    documents = documents.filter((doc) => { return doc.length }); // filter empty documents

    console.log('docs length = '+documents.length);


    console.log('Start Prediction...');
    tng_prediction.configure(documents,vocab,W, 10, 2000, 100, 10, randomSeed);
    console.log('Start running left-to-right algorithm...');
    result = tng_prediction.predict_next_word(T, alpha, beta, gamma, delta,
      n_zw, m_zwv, p_zwk, n_z, m_zw
    );
  }
  return result;
}

function makeArray(x) {
    var a = new Array();    
    for (var i=0;i<x;i++)  {
        a[i]=0;
    }
    return a;
}

function make2DArray(x,y) {
    var a = new Array();    
    for (var i=0;i<x;i++)  {
        a[i]=new Array();
        for (var j=0;j<y;j++)
            a[i][j]=0;
    }
    return a;
}

function make3DArray(x,y,z) {
  var a = new Array();    
  for (var i=0;i<x;i++)  {
      a[i]=new Array();
      for (var j=0;j<y;j++) {
          a[i][j]=new Array();
          for (var k=0;k<z;k++) {
            a[i][j][k]=0;
          }            
      }
  }
  return a;
}

var tng_prediction = new function() {
    // model state variables
    var documents; 
    var T; // # of Topic
    var D; // # of Docs
    var W; // # of unique words

    // model hyper-priors
    var alpha, beta, gamma, delta;
    
    // .. temporary variable to speedup Gibbs sampling
    var n_zw,m_zwv,p_zwk,n_z,m_zw;        
    
    var THIN_INTERVAL = 20;
    var BURN_IN = 100;
    var ITERATIONS = 1000;
    var SAMPLE_LAG;
    var RANDOM_SEED;
    var dispcol = 0;
    var numstats=0;
    var vocab = [];
    this.configure = function (docs,vocab,w,iterations,burnIn,thinInterval,sampleLag,randomSeed
    ) {
        this.ITERATIONS = iterations;
        this.BURN_IN = burnIn;
        this.THIN_INTERVAL = thinInterval;
        this.SAMPLE_LAG = sampleLag;
        this.RANDOM_SEED = randomSeed;
        this.documents = docs;
        this.W = w;
        this.D = docs.length;
        this.dispcol=0;
        this.numstats=0;
        this.vocab = vocab;         
    }
    this.initialState = function (n_zw, m_zwv, p_zwk, n_z, m_zw) {
      this.n_zw = n_zw;
      this.m_zwv = m_zwv;
      this.p_zwk = p_zwk;
      this.n_z = n_z;
      this.m_zw = m_zw;
    }
    
    this.predict_next_word = function (T, alpha, beta, gamma, delta, 
      n_zw, m_zwv, p_zwk, n_z, m_zw) {
        var i;
        this.T = T;
        this.alpha = alpha;
        this.beta = beta;
        this.gamma = gamma;
        this.delta = delta;
        
        this.initialState(n_zw, m_zwv, p_zwk, n_z, m_zw);
        
        var logNumParticles = Math.log(this.ITERATIONS);
        var totalLogLikelihood = 0;
        var tokenCount = 0;

        var m = 0;
        console.log('Processing doc #'+m);
        tokenCount += this.documents[m].length;

        var str = this.documents[m].reduce((acc, val)=>{
          return acc + ' ' + this.vocab[val];
        }, '');
        console.log(str);
        
        var sampling = this.left_to_right_sampling(m);
        return sampling;
    };

    this.left_to_right_sampling = function(m) {
      var docLength = this.documents[m].length;
      var topicAssignments = makeArray(docLength);
      var bigramAssignments = makeArray(docLength);
      
      // Create copy of stat counters for each iteration to a document
      // so it gets cleanup after evaluate each document.
      var z = makeArray(docLength);
      var x = makeArray(docLength);
      for(var i=0;i<docLength;i++) {
        z[i] = null;
        x[i] = null;
      } // for i

      // .. temporary variable to speedup Gibbs sampling
      var n_zw = make2DArray(this.T,this.W); // # of time word w is assigned to topic z as unigram
      var m_zwv = make3DArray(this.T,this.W,this.W); // # of time word v is assigned to topic z as 2nd term of word w
      var p_zwk = make3DArray(this.T,this.W,2); // # of time bigram status k is assigned for previous word w of topic (of previous word) z
      var q_dz = make2DArray(this.D,this.T); // # of word with topic z in document d
      var n_z = makeArray(this.T); // # of times any token is assigned to topic z as unigram
      var m_zw = make2DArray(this.T,this.W); // # of time any token is assigned to topic z as 2nd term of word w
      var N_d = makeArray(this.D);

      for (var t = 0; t < this.T; t++) {
        n_z[t] = this.n_z[t];
        for (var w = 0; w < this.W; w++) {
          n_zw[t][w] = this.n_zw[t][w];
          p_zwk[t][w][0] = this.p_zwk[t][w][0];
          p_zwk[t][w][1] = this.p_zwk[t][w][1];
          m_zw[t][w] = this.m_zw[t][w];
          for (var v = 0; v < this.W; v++) {
            m_zwv[t][w][v] = this.m_zwv[t][w][v];
          }
        }
      }
      
      var limit = docLength;

      for(var position=0;position<limit;position++) {
        // Disregard words those are not in dictionary.
        //console.log('w = '+this.documents[m][position]);
        if(this.documents[m][position] == -1)
          continue;
        //console.log('x = '+JSON.stringify(x));
        //console.log('re-sample on position: '+position+', word='+this.vocab[this.documents[m][position]]);          
        var sampling = this.sampleFullConditional(m, position, true,
          n_zw,m_zwv,p_zwk,q_dz,n_z,m_zw,N_d,z,x
        );  
        topicAssignments[position] = sampling.topic;                   
        bigramAssignments[position] = sampling.bigramStatus;                   
      }

      var sampling = this.predictNextWord(m,
        n_zw,m_zwv,p_zwk,q_dz,n_z,m_zw,N_d,z,x
      );
      sampling.topics = topicAssignments;
      sampling.bigrams = bigramAssignments;
      
      // console.log(' : Sampling:');
      var str = this.documents[m].reduce((acc, val, index)=>{
        if(index <= limit)
          return acc + ' ' + this.vocab[val] + '('+((bigramAssignments[index]==0)?topicAssignments[index]:'-')+')';
        else
          return acc;
      }, '');
      console.log(' : ' + str + ' ['+this.vocab[sampling.word]+'][PROB = ' + (sampling.prob[sampling.word]*100).toFixed(2) +'%]');
      return sampling;
    };

    this.sampleFullConditional = function(m,n, is_resampling,
        n_zw,m_zwv,p_zwk,q_dz,n_z,m_zw,N_d,z,x
      ) {
        var wordProbabilities = 0;
        var word = null;
        var topic = null;
        var status = null;
        var prev_topic = null;
        var prev_word = null;
        var next_status = null;

        word = this.documents[m][n];
        if(n > 0) {
          // Loop until found available previous
          for(var i=n-1;i>=0;i--) {
            if(this.documents[m][i] != -1) {
              prev_word = this.documents[m][i];
              prev_topic = z[i];        
              break;                
            }
          }          
        }
        if(n < N_d[m] - 1) {
          next_status = x[n+1]; //this.x_d_i[d][n+1];          
        }

        console.log(((is_resampling)?'RESAMPLING':'SAMPLING')+' doc['+m+']['+n+']: '+
          ((prev_word==null)?'[]':('['+this.vocab[prev_word]+']'))+
          this.vocab[word]+
          ((next_status==null)?'[]':('['+this.vocab[this.documents[m][n+1]]+']'))
        );
        console.log('topic='+JSON.stringify(z));
        console.log('status='+JSON.stringify(x));
        /*
        if(is_resampling) {
          topic = z[n];
          status = x[n];
    
          if(status == 0) {
            n_zw[topic][word]--;
            n_z[topic]--;        
          } else {
            m_zwv[topic][prev_word][word]--;
            m_zw[topic][prev_word]--;
          } // if
          if(prev_topic != null) {            
            p_zwk[prev_topic][prev_word][status]--;
            //console.log(' - resamp(1): p_zwk['+prev_topic+','+prev_word+','+status+']: '
            //  +(p_zwk[prev_topic][prev_word][status]+1)+' => '+(p_zwk[prev_topic][prev_word][status]) );
          } // if
          if(next_status != null) {
            p_zwk[topic][word][next_status]--;
            //console.log(' - resamp(2): p_zwk['+topic+','+word+','+next_status+']: '
            //  +(p_zwk[topic][word][next_status]+1)+' => '+(p_zwk[topic][word][next_status]) );          
          }
          q_dz[m][topic]--;
          //console.log('resamp - q_dz['+m+']['+topic+'] '+(q_dz[m][topic]+1)+' => '+q_dz[m][topic]);
        }
        */

        // calculate each P(z,x)
        var P_zx = make2DArray(this.T, 2);
        for(var _z=0;_z<this.T;_z++) {
          for(var _x=0;_x<2;_x++) {         
            // Skip case of bigram status for first token in document because it is invalid
            if((n == 0 && _x == 1) || (prev_word == null && _x == 1))
              continue;
            // increase counter for topic, status of current word in which calculation based on.
            if(_x == 0) {
              n_zw[_z][word]++;
              n_z[_z]++;        
            } else {
              m_zwv[_z][prev_word][word]++;
              m_zw[_z][prev_word]++;
            } // if
            if(prev_topic != null) {
              p_zwk[prev_topic][prev_word][_x]++;
              //console.log('  - try(1): p_zwk['+prev_topic+','+prev_word+','+_x+']: '
              //  +(p_zwk[prev_topic][prev_word][_x]-1)+' => '+(p_zwk[prev_topic][prev_word][_x]) );
            } // if
            if(next_status != null) {
              p_zwk[_z][word][next_status]++;
              //console.log('  - try(2): p_zwk['+_z+','+word+','+next_status+']: '
              //  +(p_zwk[_z][word][next_status]-1)+' => '+(p_zwk[_z][word][next_status]) );
            }              
            q_dz[m][_z]++;
            //console.log('try - q_dz['+m+']['+_z+'] '+(q_dz[m][_z]-1)+' => '+q_dz[m][_z]);
            
            /*
            console.log(
              ((x==0)?',n_zw[z][word]='+n_zw[z][word]:',m_zwv[z][prev_word][word]='+m_zwv[z][prev_word][word])+
              ((x==0)?',n_z[z]='+n_z[z]:',n_zw[z][prev_word]='+n_zw[z][prev_word])+
              ((prev_topic != null)?',p_zwk[prev_topic][prev_word][x]='+p_zwk[prev_topic][prev_word][x]:'')+
              ((next_status != null)?',p_zwk[z][word][next_status]='+p_zwk[z][word][next_status]:'')+
              ',q_dz[m][z]='+q_dz[m][z]
            );
            */
              
            var first_term = 0.0;
            //console.log('  Denom1  = '+((n+1) + this.T * this.alpha - 1));
            if(prev_topic != null) {
              //console.log('  this.gamma = ' + this.gamma);
              //console.log('  p_zwk[prev_topic][prev_word][0] = ' + p_zwk[prev_topic][prev_word][0]);
              //console.log('  p_zwk[prev_topic][prev_word][1] = ' + p_zwk[prev_topic][prev_word][1]);
              //console.log('  Denom2  = '+(2 * this.gamma + p_zwk[prev_topic][prev_word][0] + p_zwk[prev_topic][prev_word][1] - 1));
              first_term = (this.gamma + p_zwk[prev_topic][prev_word][_x] - 1)
                * (this.alpha + q_dz[m][_z] - 1) / (
                (2 * this.gamma + p_zwk[prev_topic][prev_word][0] + p_zwk[prev_topic][prev_word][1] - 1)
                * ((n+1) + this.T * this.alpha - 1)
                );
              //console.log('ft[0]: '+first_term+', this.p_zwk[prev_topic][prev_word][x] = '+this.p_zwk[prev_topic][prev_word][x]);
            } else {
              first_term = (this.gamma)
                * (this.alpha + q_dz[m][_z] - 1) / (
                (this.gamma)
                * ((n+1) + this.T * this.alpha - 1)
                );
              //console.log('ft[1]: '+first_term);
            }
            //if(!is_resampling) {
            //  console.log('  First Term  = '+first_term);
            //}
  
            var second_term = null;
            if(_x == 0) {
              second_term = 
                (this.beta + n_zw[_z][word] - 1)
                / (this.W * this.beta + n_z[_z] - 1);
            } else {
              second_term = 
                (this.delta + m_zwv[_z][prev_word][word] - 1)
                / (this.W * this.delta + m_zw[_z][prev_word] - 1);
            }
            //if(!is_resampling) {
            //  console.log('  Second Term  = '+second_term);
            //}

            //console.log('first_term = '+first_term+', second_term = '+second_term);
            P_zx[_z][_x] = first_term * second_term;
            wordProbabilities += P_zx[_z][_x];
            
            // decrease counter back.
            if(_x == 0) {
              n_zw[_z][word]--;
              n_z[_z]--;        
            } else {
              m_zwv[_z][prev_word][word]--;
              m_zw[_z][prev_word]--;
            } // if
            if(prev_topic != null) {
              p_zwk[prev_topic][prev_word][_x]--;
              //console.log('  - roll(1): p_zwk['+prev_topic+','+prev_word+','+_x+']: '
              //  +(p_zwk[prev_topic][prev_word][_x]+1)+' => '+(p_zwk[prev_topic][prev_word][_x]) );
            } // if
            if(next_status != null) {
              p_zwk[_z][word][next_status]--;
              //console.log('  - roll(2): p_zwk['+_z+','+word+','+next_status+']: '
              //  +(p_zwk[_z][word][next_status]+1)+' => '+(p_zwk[_z][word][next_status]) );
          }
            q_dz[m][_z]--;            
            //console.log('roll - q_dz['+m+']['+_z+'] '+(q_dz[m][_z]+1)+' => '+q_dz[m][_z]);            
          } // for x
        } // for z
            
        // Sampling new topic, status from calculated P_zx
        // Note that if (token == 0) then x is forced to be 0...
        // Sampling in such case has to ignore case of x=1 and sampling from the rest probability.
        //console.log('Sampling: Doc['+document+']['+token+']');
        var sum = 0;
        for(var _z=0;_z<this.T;_z++) {
          for(var _x=0;_x<2;_x++) {          
            sum = sum + P_zx[_z][_x];
            P_zx[_z][_x] = sum;
            //console.log(' - acc P_zx['+_z+']['+_x+'] = '+P_zx[_z][_x]);
          } // for x
        } // for z
        var u = this.getRandom() * sum;
        var new_topic = null;
        var new_status = null;
        for(var _z=0;_z<this.T;_z++) {
          for(var _x=0;_x<2;_x++) {
            //console.log('[z][x]='+_z+',',x+': p='+P_zx[_z][_x]);          
            if(u < P_zx[_z][_x]) {
              new_topic = _z;
              new_status = _x;
              break;
            } // if
          } // for x
          if(new_topic != null)
            break;
        } // for z
        topic = new_topic;
        status = new_status;

        z[n] = topic;
        x[n] = new_status;
        
        // Update all data, counter based on our sampling result
        if(status == 0) {
          n_zw[topic][word]++;
          n_z[topic]++;        
        } else {
          //console.log('topic='+topic+',prevw='+prev_word+',w='+word);
          m_zwv[topic][prev_word][word]++;
          m_zw[topic][prev_word]++;
        } // if
        if(prev_topic != null) {
          p_zwk[prev_topic][prev_word][status]++;
          //console.log('  - set(1): p_zwk['+prev_topic+','+prev_word+','+status+']: '
          //  +(p_zwk[prev_topic][prev_word][status]-1)+' => '+(p_zwk[prev_topic][prev_word][status]) );
        } // if
        if(next_status != null) {
          p_zwk[topic][word][next_status]++;
          //console.log('  - set(2): p_zwk['+topic+','+word+','+next_status+']: '
          //  +(p_zwk[topic][word][next_status]-1)+' => '+(p_zwk[topic][word][next_status]) );
      }
        q_dz[m][topic]++;        
        //console.log('set - q_dz['+m+']['+topic+'] '+(q_dz[m][topic]-1)+' => '+q_dz[m][topic]);            
        
        N_d[m]++;
        return {topic: topic, prob: wordProbabilities, bigramStatus: status};
    }

    this.predictNextWord = function(m,
      n_zw,m_zwv,p_zwk,q_dz,n_z,m_zw,N_d,z,x
    ) {
      var word = 0;
      var topic = null;
      var status = null;
      var prev_topic = null;
      var prev_word = null;
      var next_status = null;

      if(this.documents[m].length > 0) {
        // Loop until found available previous
        for(var i=this.documents[m].length-1;i>=0;i--) {
          if(this.documents[m][i] != -1) {
            prev_word = this.documents[m][i];
            prev_topic = z[i];        
            break;                
          }
        }
      }

      //console.log(((is_resampling)?'RESAMPLING':'SAMPLING')+' doc['+m+']['+n+']: '+
      //  ((prev_word==null)?'[]':('['+this.vocab[prev_word]+']'))+
      //  this.vocab[word]+
      //  ((next_status==null)?'[]':('['+this.vocab[this.documents[m][n+1]]+']'))
      //);
    
      // calculate each P(z,x)
      var p = makeArray(this.W);
      for(var _z=0;_z<this.T;_z++) {
        for(var _x=0;_x<2;_x++) {     
          for (var w = 0; w < this.W; w++) {    
            // Skip case of bigram status for first token in document because it is invalid
            if((this.documents[m].length == 0 && _x == 1) || ( prev_word == null && _x == 1))
              continue;
            // increase counter for topic, status of current word in which calculation based on.
            if(_x == 0) {
              n_zw[_z][w]++;
              n_z[_z]++;        
            } else {
              m_zwv[_z][prev_word][w]++;
              m_zw[_z][prev_word]++;
            } // if
            if(prev_topic != null) {
              p_zwk[prev_topic][prev_word][_x]++;
              //console.log('  - try(1): p_zwk['+prev_topic+','+prev_word+','+_x+']: '
              //  +(p_zwk[prev_topic][prev_word][_x]-1)+' => '+(p_zwk[prev_topic][prev_word][_x]) );
            } // if
            if(next_status != null) {
              p_zwk[_z][w][next_status]++;
              //console.log('  - try(2): p_zwk['+_z+','+word+','+next_status+']: '
              //  +(p_zwk[_z][word][next_status]-1)+' => '+(p_zwk[_z][word][next_status]) );
            }              
            q_dz[m][_z]++;
            //console.log('try - q_dz['+m+']['+_z+'] '+(q_dz[m][_z]-1)+' => '+q_dz[m][_z]);
            
            /*
            console.log(
              ((x==0)?',n_zw[z][word]='+n_zw[z][word]:',m_zwv[z][prev_word][word]='+m_zwv[z][prev_word][word])+
              ((x==0)?',n_z[z]='+n_z[z]:',n_zw[z][prev_word]='+n_zw[z][prev_word])+
              ((prev_topic != null)?',p_zwk[prev_topic][prev_word][x]='+p_zwk[prev_topic][prev_word][x]:'')+
              ((next_status != null)?',p_zwk[z][word][next_status]='+p_zwk[z][word][next_status]:'')+
              ',q_dz[m][z]='+q_dz[m][z]
            );
            */
              
            var first_term = 0.0;
            //console.log('  Denom1  = '+((n+1) + this.T * this.alpha - 1));
            if(prev_topic != null) {
              //console.log('  this.gamma = ' + this.gamma);
              //console.log('  p_zwk[prev_topic][prev_word][0] = ' + p_zwk[prev_topic][prev_word][0]);
              //console.log('  p_zwk[prev_topic][prev_word][1] = ' + p_zwk[prev_topic][prev_word][1]);
              //console.log('  Denom2  = '+(2 * this.gamma + p_zwk[prev_topic][prev_word][0] + p_zwk[prev_topic][prev_word][1] - 1));
              first_term = (this.gamma + p_zwk[prev_topic][prev_word][_x] - 1)
                * (this.alpha + q_dz[m][_z] - 1) / (
                (2 * this.gamma + p_zwk[prev_topic][prev_word][0] + p_zwk[prev_topic][prev_word][1] - 1)
                * ((this.documents[m].length+1) + this.T * this.alpha - 1)
                );
              //console.log('ft[0]: '+first_term+', this.p_zwk[prev_topic][prev_word][x] = '+this.p_zwk[prev_topic][prev_word][x]);
            } else {
              first_term = (this.gamma)
                * (this.alpha + q_dz[m][_z] - 1) / (
                (this.gamma)
                * ((this.documents[m].length+1) + this.T * this.alpha - 1)
                );
              //console.log('ft[1]: '+first_term);
            }
            //if(!is_resampling) {
            //  console.log('  First Term  = '+first_term);
            //}

            var second_term = null;
            if(_x == 0) {
              second_term = 
                (this.beta + n_zw[_z][w] - 1)
                / (this.W * this.beta + n_z[_z] - 1);
            } else {
              second_term = 
                (this.delta + m_zwv[_z][prev_word][w] - 1)
                / (this.W * this.delta + m_zw[_z][prev_word] - 1);
            }
            //if(!is_resampling) {
            //  console.log('  Second Term  = '+second_term);
            //}

            //console.log('first_term = '+first_term+', second_term = '+second_term);
            p[w] += first_term * second_term;
            
            // decrease counter back.
            if(_x == 0) {
              n_zw[_z][w]--;
              n_z[_z]--;        
            } else {
              m_zwv[_z][prev_word][w]--;
              m_zw[_z][prev_word]--;
            } // if
            if(prev_topic != null) {
              p_zwk[prev_topic][prev_word][_x]--;
              //console.log('  - roll(1): p_zwk['+prev_topic+','+prev_word+','+_x+']: '
              //  +(p_zwk[prev_topic][prev_word][_x]+1)+' => '+(p_zwk[prev_topic][prev_word][_x]) );
            } // if
            if(next_status != null) {
              p_zwk[_z][w][next_status]--;
              //console.log('  - roll(2): p_zwk['+_z+','+word+','+next_status+']: '
              //  +(p_zwk[_z][word][next_status]+1)+' => '+(p_zwk[_z][word][next_status]) );
            }
            q_dz[m][_z]--;            
            //console.log('roll - q_dz['+m+']['+_z+'] '+(q_dz[m][_z]+1)+' => '+q_dz[m][_z]);            
          } // for w
        } // for x
      } // for z
          
      // Sampling new topic, status from calculated P_zx
      // Note that if (token == 0) then x is forced to be 0...
      // Sampling in such case has to ignore case of x=1 and sampling from the rest probability.
      //console.log('Sampling: Doc['+document+']['+token+']');

      var sorted_p = [];        
      for (var k = 0; k < p.length; k++) {
        sorted_p.push({word:k, prob:p[k]})
      }
      sorted_p.sort(function(a, b) {
        if (a.prob < b.prob)
          return 1;
        if (a.prob > b.prob)
          return -1;
        return 0;          
      });
      console.log(JSON.stringify(sorted_p));
      return {word: sorted_p[0].word, prob: p, sort_p: sorted_p};    
    }
  
    this.getRandom = function() {
        if (this.RANDOM_SEED) {
            // generate a pseudo-random number using a seed to ensure reproducable results.
            var x = Math.sin(this.RANDOM_SEED++) * 1000000;
            return x - Math.floor(x);
        } else {
            // use standard random algorithm.
            return Math.random();
        }
    }
}

module.exports = process;
