/*
Inference for Llama-2 Transformer model in pure C.

Example compile: (see README for more details)
$ gcc -O3 -o run run.c -lm

Then run with:
$ ./run
*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <ctype.h>
#include <stdint.h>

// ----------------------------------------------------------------------------
// Transformer and RunState structs, and related memory management

typedef struct {
    int dim; // transformer dimension
    int hidden_dim; // for ffn layers
    int n_layers; // number of layers
    int n_heads; // number of query heads
    int n_kv_heads; // number of key/value heads (can be < query heads because of multiquery)
    int vocab_size; // vocabulary size, usually 256 (byte-level)
    int seq_len; // max sequence length
} Config;

typedef struct {
    // token embedding table
    float* token_embedding_table;    // (vocab_size, dim)
    // weights for rmsnorms
    float* rms_att_weight; // (layer, dim) rmsnorm weights
    float* rms_ffn_weight; // (layer, dim)
    // weights for matmuls
    float* wq; // (layer, dim, dim)
    float* wk; // (layer, dim, dim)
    float* wv; // (layer, dim, dim)
    float* wo; // (layer, dim, dim)
    // weights for ffn
    float* w1; // (layer, hidden_dim, dim)
    float* w2; // (layer, dim, hidden_dim)
    float* w3; // (layer, hidden_dim, dim)
    // final rmsnorm
    float* rms_final_weight; // (dim,)
    // freq_cis for RoPE relatively positional embeddings
    float* freq_cis_real; // (seq_len, dim/2)
    float* freq_cis_imag; // (seq_len, dim/2)
    // (optional) classifier weights for the logits, on the last layer
    float* wcls;
} TransformerWeights;

typedef struct {
    // current wave of activations
    float *x; // activation at current time stamp (dim,)
    float *xb; // same, but inside a residual branch (dim,)
    float *xb2; // an additional buffer just for convenience (dim,)
    float *hb; // buffer for hidden dimension in the ffn (hidden_dim,)
    float *hb2; // buffer for hidden dimension in the ffn (hidden_dim,)
    float *q; // query (dim,)
    float *k; // key (dim,)
    float *v; // value (dim,)
    float *att; // buffer for scores/attention values (n_heads, seq_len)
    float *logits; // output logits
    // kv cache
    float* key_cache;   // (layer, seq_len, dim)
    float* value_cache; // (layer, seq_len, dim)
} RunState;

void malloc_run_state(RunState* s, Config* p) {
    // we calloc instead of malloc to keep valgrind happy
    s->x = calloc(p->dim, sizeof(float));
    s->xb = calloc(p->dim, sizeof(float));
    s->xb2 = calloc(p->dim, sizeof(float));
    s->hb = calloc(p->hidden_dim, sizeof(float));
    s->hb2 = calloc(p->hidden_dim, sizeof(float));
    s->q = calloc(p->dim, sizeof(float));
    s->k = calloc(p->dim, sizeof(float));
    s->v = calloc(p->dim, sizeof(float));
    s->att = calloc(p->n_heads * p->seq_len, sizeof(float));
    s->logits = calloc(p->vocab_size, sizeof(float));
    s->key_cache = calloc(p->n_layers * p->seq_len * p->dim, sizeof(float));
    s->value_cache = calloc(p->n_layers * p->seq_len * p->dim, sizeof(float));
    // ensure all mallocs went fine
    if (!s->x || !s->xb || !s->xb2 || !s->hb || !s->hb2 || !s->q 
     || !s->k || !s->v || !s->att || !s->logits || !s->key_cache 
     || !s->value_cache) {
        printf("malloc failed!\n");
        exit(1);
    }
}

void free_run_state(RunState* s) {
    free(s->x);
    free(s->xb);
    free(s->xb2);
    free(s->hb);
    free(s->hb2);
    free(s->q);
    free(s->k);
    free(s->v);
    free(s->att);
    free(s->logits);
    free(s->key_cache);
    free(s->value_cache);
}

// ----------------------------------------------------------------------------
// initialization: read from checkpoint

void checkpoint_init_weights(TransformerWeights *w, Config* p, float* f, int shared_weights) {
    float* ptr = f;
    w->token_embedding_table = ptr;
    ptr += p->vocab_size * p->dim;
    w->rms_att_weight = ptr;
    ptr += p->n_layers * p->dim;
    w->wq = ptr;
    ptr += p->n_layers * p->dim * p->dim;
    w->wk = ptr;
    ptr += p->n_layers * p->dim * p->dim;
    w->wv = ptr;
    ptr += p->n_layers * p->dim * p->dim;
    w->wo = ptr;
    ptr += p->n_layers * p->dim * p->dim;
    w->rms_ffn_weight = ptr;
    ptr += p->n_layers * p->dim;
    w->w1 = ptr;
    ptr += p->n_layers * p->dim * p->hidden_dim;
    w->w2 = ptr;
    ptr += p->n_layers * p->hidden_dim * p->dim;
    w->w3 = ptr;
    ptr += p->n_layers * p->dim * p->hidden_dim;
    w->rms_final_weight = ptr;
    ptr += p->dim;
    w->freq_cis_real = ptr;
    int head_size = p->dim / p->n_heads;
    ptr += p->seq_len * head_size / 2;
    w->freq_cis_imag = ptr;
    ptr += p->seq_len * head_size / 2;
    w->wcls = shared_weights ? w->token_embedding_table : ptr;
}

// ----------------------------------------------------------------------------
// neural net blocks

void accum(float *a, float *b, int size) {
    for (int i = 0; i < size; i++) {
        a[i] += b[i];
    }
}

void rmsnorm(float* o, float* x, float* weight, int size) {
    // calculate sum of squares
    float ss = 0.0f;
    for (int j = 0; j < size; j++) {
        ss += x[j] * x[j];
    }
    ss /= size;
    ss += 1e-5f;
    ss = 1.0f / sqrtf(ss);
    // normalize and scale
    for (int j = 0; j < size; j++) {
        o[j] = weight[j] * (ss * x[j]);
    }
}

void softmax(float* x, int size) {
    // find max value (for numerical stability)
    float max_val = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }
    // exp and sum
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    // normalize
    for (int i = 0; i < size; i++) {
        x[i] /= sum;
    }
}

void matmul(float* xout, float* x, float* w, int n, int d) {
    // W (d,n) @ x (n,) -> xout (d,)
    #pragma omp parallel for
    for (int i = 0; i < d; i++) {
        float val = 0.0f;
        for (int j = 0; j < n; j++) {
            val += w[i * n + j] * x[j];
        }
        xout[i] = val;
    }
}

void transformer(int token, int pos, Config* p, RunState* s, TransformerWeights* w) {
    
    // a few convenience variables
    float *x = s->x;
    int dim = p->dim;
    int hidden_dim =  p->hidden_dim;
    int head_size = dim / p->n_heads;

    // copy the token embedding into x
    float* content_row = &(w->token_embedding_table[token * dim]);
    memcpy(x, content_row, dim*sizeof(*x));

    // pluck out the "pos" row of freq_cis_real and freq_cis_imag
    float* freq_cis_real_row = w->freq_cis_real + pos * head_size / 2;
    float* freq_cis_imag_row = w->freq_cis_imag + pos * head_size / 2;

    // forward all the layers
    for(int l = 0; l < p->n_layers; l++) {
    
        // attention rmsnorm
        rmsnorm(s->xb, x, w->rms_att_weight + l*dim, dim);

        // qkv matmuls for this position
        matmul(s->q, s->xb, w->wq + l*dim*dim, dim, dim);
        matmul(s->k, s->xb, w->wk + l*dim*dim, dim, dim);
        matmul(s->v, s->xb, w->wv + l*dim*dim, dim, dim);

        // apply RoPE rotation to the q and k vectors for each head
        for (int h = 0; h < p->n_heads; h++) {
            // get the q and k vectors for this head
            float* q = s->q + h * head_size;
            float* k = s->k + h * head_size;
            // rotate q and k by the freq_cis_real and freq_cis_imag
            for (int i = 0; i < head_size; i+=2) {
                float q0 = q[i];
                float q1 = q[i+1];
                float k0 = k[i];
                float k1 = k[i+1];
                float fcr = freq_cis_real_row[i/2];
                float fci = freq_cis_imag_row[i/2];
                q[i]   = q0 * fcr - q1 * fci;
                q[i+1] = q0 * fci + q1 * fcr;
                k[i]   = k0 * fcr - k1 * fci;
                k[i+1] = k0 * fci + k1 * fcr;
            }
        }

        // save key,value at this time step (pos) to our kv cache
        int loff = l * p->seq_len * dim; // kv cache layer offset for convenience
        float* key_cache_row = s->key_cache + loff + pos * dim;
        float* value_cache_row = s->value_cache + loff + pos * dim;
        memcpy(key_cache_row, s->k, dim*sizeof(*key_cache_row));
        memcpy(value_cache_row, s->v, dim*sizeof(*value_cache_row));
        
        // multihead attention. iterate over all heads
        #pragma omp parallel for
        for (int h = 0; h < p->n_heads; h++) {
            // get the query vector for this head
            float* q = s->q + h * head_size;
            // attention scores for this head
            float* att = s->att + h * p->seq_len;
            // iterate over all timesteps, including the current one
            for (int t = 0; t <= pos; t++) {
                // get the key vector for this head and at this timestep
                float* k = s->key_cache + loff + t * dim + h * head_size;
                // calculate the attention score as the dot product of q and k
                float score = 0.0f;
                for (int i = 0; i < head_size; i++) {
                    score += q[i] * k[i];
                }
                score /= sqrtf(head_size);
                // save the score to the attention buffer
                att[t] = score;
            }

            // softmax the scores to get attention weights, from 0..pos inclusively
            softmax(att, pos + 1);
            
            // weighted sum of the values, store back into xb
            for (int i = 0; i < head_size; i++) {
                float val = 0.0f;
                for (int t = 0; t <= pos; t++) {
                    val += att[t] * s->value_cache[loff + t * dim + h * head_size + i]; // note bad locality
                }
                s->xb[h * head_size + i] = val;
            }
        }

        // final matmul to get the output of the attention
        matmul(s->xb2, s->xb, w->wo + l*dim*dim, dim, dim);

        // residual connection back into x
        accum(x, s->xb2, dim);

        // ffn rmsnorm
        rmsnorm(s->xb, x, w->rms_ffn_weight + l*dim, dim);

        // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
        // first calculate self.w1(x) and self.w3(x)
        matmul(s->hb, s->xb, w->w1 + l*dim*hidden_dim, dim, hidden_dim);
        matmul(s->hb2, s->xb, w->w3 + l*dim*hidden_dim, dim, hidden_dim);
        
        // F.silu; silu(x)=x*σ(x),where σ(x) is the logistic sigmoid
        for (int i = 0; i < hidden_dim; i++) {
            s->hb[i] = s->hb[i] * (1.0f / (1.0f + expf(-s->hb[i])));
        }
        
        // elementwise multiply with w3(x)
        for (int i = 0; i < hidden_dim; i++) {
            s->hb[i] = s->hb[i] * s->hb2[i];
        }

        // final matmul to get the output of the ffn
        matmul(s->xb, s->hb, w->w2 + l*dim*hidden_dim, hidden_dim, dim);

        // residual connection
        accum(x, s->xb, dim);
    }
    
    // final rmsnorm
    rmsnorm(x, x, w->rms_final_weight, dim);

    // classifier into logits
    matmul(s->logits, x, w->wcls, p->dim, p->vocab_size);
}

int sample(float* probabilities, int n) {
    // sample index from probabilities, they must sum to 1
    float r = (float)rand() / (float)RAND_MAX;
    float cdf = 0.0f;
    for (int i = 0; i < n; i++) {
        cdf += probabilities[i];
        if (r < cdf) {
            return i;
        }
    }
    return n - 1; // in case of rounding errors
}

int argmax(float* v, int n) {
    // return argmax of v in elements 0..n
    int max_i = 0;
    float max_p = v[0];
    for (int i = 1; i < n; i++) {
        if (v[i] > max_p) {
            max_i = i;
            max_p = v[i];
        }
    }
    return max_i;
}

// ----------------------------------------------------------------------------

long time_in_ms() {
    struct timespec time;
    // Get the current time with nanosecond precision
    if (clock_gettime(CLOCK_REALTIME, &time) == 0) {
        return time.tv_sec * 1000 + time.tv_nsec / 1000000;
    } else {
        perror("clock_gettime");
        return -1; // Return -1 to indicate an error
    }
}

char* create_word(const char* token, int postfix_ed, int postfix_ing, int postfix_s)
{
    static char buffer[1024];
    strcpy(buffer, token);
    char* end = buffer + strlen(buffer);
    if (postfix_s) {
        if (end[-1] == 's' || end[-1] == 'x' || end[-1] == 'z') {
            *end++ = 'e';
        }
        if (end[-1] == 'h' && (end[-2] == 's' || end[-2] == 'c')) {
            *end++ = 'e';
        }
        *end++ = 's';
    }
    if (postfix_ed) {
        if (end[-1] != 'e') {
            *end++ = 'e';
        }
        *end++ = 'd';
    }
    if (postfix_ing) {
        if (end[-1] == 'e') {
            end--;
        }
        *end++ = 'i';
        *end++ = 'n';
        *end++ = 'g';
    }
    *end++ = 0;
    return buffer;
}

static char prev[1024] = "";
static char this[1024];
static int capitalize = 0;
static int uppercase = 0;
static int postfix_ed = 0;
static int postfix_ing = 0;
static int postfix_s = 0;

//#define LOG(format, ...) fprintf(stderr, "\n\e[1;31m" format "\e[0m\n", ##__VA_ARGS__)
#define LOG(format, ...)

char* decode_token(const char* token) {

    LOG("---------------------------------------------\ntoken: %s", token);

    static char buffer[1024];
    char* end = buffer;

    if (token[0] >= '0' && token[0] <= '9') {
        LOG("digit");
        strcpy(this, "0");
    } else if (isalpha(token[0])) {
        LOG("word");
        strcpy(this, "a");
    } else {
        LOG("other");
        strcpy(this, token);
    }
    
    if (strcmp("~1", token) == 0) {
        LOG("capitalize");
        capitalize = 1;
    } else if (strcmp("~2", token) == 0) {
        LOG("uppercase");
        uppercase = 1;
    } else if (strcmp("-s", token) == 0) {
        LOG("postfix_s");
        postfix_s = 1;
    } else if (strcmp("-ed", token) == 0) {
        LOG("postfix_ed");
        postfix_ed = 1;
    } else if (strcmp("-ing", token) == 0) {
        LOG("postfix_ing");
        postfix_ing = 1;
    } else {
        char x;
        if (prev[0] == 0 || strcmp(this, "'>") == 0 || strcmp(this, "\">") == 0) {
            //pass
            x = '1';
        } else if (strcmp(prev, "--") == 0) {
            *end++ = ' ';
            x = '2';
        } else if (strcmp(prev, "'<") == 0 || strcmp(prev, "\"<") == 0 || (this[1] == 0 && (this[0] == '-' || this[0] == '\'' || this[0] == '.' || this[0] == ',' || this[0] == '!' || this[0] == '?' || this[0] == ':'))) {
            //pass
            x = '3';
        } else if (strcmp(prev, "'>") == 0 || strcmp(prev, "\">") == 0 || (prev[1] == 0 && (prev[0] == 'a' || prev[0] == '.' || prev[0] == ',' || prev[0] == '!' || prev[0] == '?' || prev[0] == ':')) || strcmp(this, "--") == 0) {
            *end++ = ' ';
            x = '4';
        } else if ((prev[1] == 0 && (prev[0] == '\'' || prev[0] == '-')) || strcmp(this, "0") == 0) {
            //pass
            x = '5';
        } else {
            *end++ = ' ';
            x = '6';
        }
        strcpy(prev, this);

        if (strcmp(prev, "'")) {
            LOG("----------------%c", x);
        }

        if (strcmp(this, "a") == 0) {
            char* word = create_word(token, postfix_ed, postfix_ing, postfix_s);
            if (capitalize) {
                word[0] = toupper(word[0]);
            }
            if (uppercase) {
                int i;
                for (i = 0; i < strlen(word); i++) {
                    word[i] = toupper(word[i]);
                }
            }
            strcpy(end, word);
            end += strlen(word);
            LOG("Word: %s", word);
        } else if (strcmp("0", this) ||
                    strcmp(".", this) == 0 ||
                    strcmp(",", this) == 0 ||
                    strcmp("'", this) == 0 ||
                    strcmp("!", this) == 0 ||
                    strcmp("?", this) == 0 ||
                    strcmp("-", this) == 0 ||
                    strcmp(":", this) == 0 ||
                    strcmp("\"<", this) == 0 ||
                    strcmp("\">", this) == 0 ||
                    strcmp("--", this) == 0 ||
                    strcmp("'>", this) == 0 ||
                    strcmp("'<", this) == 0)
        {
            *end++ = token[0];
        } else {
            strcpy(end, token);
            end += strlen(token);
        }
        capitalize = 0;
        uppercase = 0;
        postfix_ed = 0;
        postfix_ing = 0;
        postfix_s = 0;
    }

    *end = 0;

    return buffer;
}

uint16_t* check_compression(float* values, int length)
{
    int i;
    static int exp_hist[256] = {0};
    static int tz_hist[24] = {0};
    uint32_t* bin_values = (uint32_t*)values;
    uint16_t* output = malloc(2 * length);
    uint16_t* output_ptr = output;
    for (i = 0; i < length; i++) {
        uint32_t value = bin_values[i];
        uint32_t negative = (value >> 31) != 0;
        uint32_t exp = (value >> 23) & 0xFF;
        uint32_t frac = value & 0x7FFFFF;
        exp_hist[exp]++;
        if (exp == 0) {
            //printf("zero: %06X\n", frac);
        }
        int tz = 0;
        uint32_t fv = frac;
        while (tz < 23 && !(fv & 1)) {
            fv >>= 1;
            tz++;
        }
        tz_hist[tz]++;
        //frac = ((frac + 16) >> 5) << 5;
        frac = ((frac + 4096) >> 13) << 13;
        if (frac > 0x7FFFFF) {
            frac = 0x7FFFFF;
        }
        bin_values[i] = (bin_values[i] & 0xFF800000) | frac;
        if (exp > 0) {
            exp = exp - 127 + 24;
        }
        uint16_t out_val = exp << 10;
        out_val |= frac >> 13;
        out_val |= negative ? 0x8000 : 0;
        *output_ptr++ = out_val;
    }
    for (i = 0; i < 256; i++) {
        //printf("exp: %4d   %6d\n", i - 127, exp_hist[i]);
    }
    for (i = 0; i < 24; i++) {
        //printf("frac tz: %4d   %6d\n", i, tz_hist[i] * (1 << i));
    }
    return output;
}

int main(int argc, char *argv[]) {

    // poor man's C argparse
    int i;
    char *checkpoint = NULL;  // e.g. out/model.bin
    float temperature = 0.9f; // e.g. 1.0, or 0.0
    int steps = 256;          // max number of steps to run for, 0: use seq_len
    // 'checkpoint' is necessary arg
    if (argc < 2) {
        printf("Usage: %s <checkpoint_file> [temperature] [steps]\n", argv[0]);
        return 1;
    }
    if (argc >= 2) {
        checkpoint = argv[1];
    }
    if (argc >= 3) {
        // optional temperature. 0.0 = (deterministic) argmax sampling. 1.0 = baseline
        temperature = atof(argv[2]);
    }
    if (argc >= 4) {
        steps = atoi(argv[3]);
    }

    // seed rng with time. if you want deterministic behavior use temperature 0.0
    srand((unsigned int)time(NULL)); 
    
    // read in the model.bin file
    Config config;
    TransformerWeights weights;
    int fd = 0;
    float* data = NULL;
    long file_size;
    {
        FILE *file = fopen(checkpoint, "rb");
        if (!file) {
            printf("Unable to open the checkpoint file %s!\n", checkpoint);
            return 1;
        }
        // read in the config header
        if(fread(&config, sizeof(Config), 1, file) != 1) { return 1; }
        // negative vocab size is hacky way of signaling unshared weights. bit yikes.
        int shared_weights = config.vocab_size > 0 ? 1 : 0;
        config.vocab_size = abs(config.vocab_size);
        // figure out the file size
        fseek(file, 0, SEEK_END); // move file pointer to end of file
        file_size = ftell(file); // get the file size, in bytes
        fclose(file);
        // memory map the Transformer weights into the data pointer
        fd = open(checkpoint, O_RDONLY); // open in read only mode
        if (fd == -1) { printf("open failed!\n"); return 1; }
        data = mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
        if (data == MAP_FAILED) { printf("mmap failed!\n"); return 1; }
        float* data2 = malloc(file_size);
        memcpy(data2, data, file_size);
        data = data2;
        float* weights_ptr = data + sizeof(Config)/sizeof(float);
        int num = (file_size - sizeof(Config)) / sizeof(float);
        uint16_t* o = check_compression(weights_ptr, num);
        checkpoint_init_weights(&weights, &config, weights_ptr, shared_weights);
        char path[1024];
        strcpy(path, checkpoint);
        strcat(path, ".16");
        file = fopen(path, "wb");
        fwrite(&config, sizeof(config), 1, file);
        fwrite(o, num * 2, 1, file);
        fclose(file);
        strcpy(path, checkpoint);
        strcat(path, ".c");
        file = fopen(path, "w");
        fprintf(file, "#define INIT_CONFIG {\\\n");
        fprintf(file, "    .dim = %d,\\\n", config.dim);
        fprintf(file, "    .hidden_dim = %d,\\\n", config.hidden_dim);
        fprintf(file, "    .n_layers = %d,\\\n", config.n_layers);
        fprintf(file, "    .n_heads = %d,\\\n", config.n_heads);
        fprintf(file, "    .n_kv_heads = %d,\\\n", config.n_kv_heads);
        fprintf(file, "    .vocab_size = %d,\\\n", config.vocab_size);
        fprintf(file, "    .seq_len = %d,\\\n", config.seq_len);
        fprintf(file, "}\n");
        fprintf(file, "static const uint16_t model_weights[%d] = {", num);
        for (i = 0; i < num; i++) {
            fprintf(file, "%s0x%04X,%s", i % 20 == 0 ? "\n    " : "", o[i], i % 20 != 19 && i != num - 1 ? " " : "");
        }
        fprintf(file, "\n};\n");
        fclose(file);
    }
    // right now we cannot run for more than config.seq_len steps
    if (steps <= 0) { steps = config.seq_len; }

    // read in the tokenizer.bin file
    char** vocab = (char**)malloc(config.vocab_size * sizeof(char*));
    {
        FILE *file = fopen("tokenizer2000.bin", "rb");
        if (!file) {
            printf("Unable to open the tokenizer file tokenizer.bin! Run "
            "python tokenizer.py to convert tokenizer.model -> tokenizer.bin\n");
            return 1;
        }
        int len;
        for (int i = 0; i < config.vocab_size; i++) {
            if(fread(&len, sizeof(int), 1, file) != 1) { return 1; }
            vocab[i] = (char *)malloc(len + 1);
            if(fread(vocab[i], len, 1, file) != 1) { return 1; }
            vocab[i][len] = '\0'; // add the string terminating token
        }
        fclose(file);
        char path[1024];
        strcpy(path, checkpoint);
        strcat(path, ".c");
        file = fopen(path, "a");
        fprintf(file, "static const char* const vocab[] = {");
        int ll = 10000;
        for (int i = 0; i < config.vocab_size; i++) {
            const char* src = vocab[i];
            char* dst = path;
            while (*src) {
                char c = *src++;
                if (c == '"' || c == '\\') {
                    *dst++ = '\\';
                }
                *dst++ = c;
            }
            *dst++ = 0;
            if (ll + strlen(path) + 4 > 120) {
                fprintf(file, "\n    ");
                ll = 4;
            }
            ll += strlen(path) + 4;
            fprintf(file, "\"%s\", ", path);
        }
        fprintf(file, "\n};\n");
        fclose(file);
    }

    // create and init the application RunState
    RunState state;
    malloc_run_state(&state, &config);
    
    // the current position we are in
    long start = time_in_ms();
    int next;
    int token = 1; // 1 = BOS token in Llama-2 sentencepiece
    int pos = 0;
    while (pos < steps || token != 1) {

        if (pos % config.seq_len == 0) {
            //printf("\n------------------\n");
        }

        // forward the transformer to get logits for the next token
        transformer(token, pos % config.seq_len, &config, &state, &weights);

        // sample the next token
        if(temperature == 0.0f) {
            // greedy argmax sampling
            next = argmax(state.logits, config.vocab_size);
        } else {
            // apply the temperature to the logits
            for (int q=0; q<config.vocab_size; q++) { state.logits[q] /= temperature; }
            // apply softmax to the logits to get the probabilities for next token
            softmax(state.logits, config.vocab_size);
            // we now want to sample from this distribution to get the next token
            next = sample(state.logits, config.vocab_size);
        }
        printf("%s", decode_token(vocab[next]));
        fflush(stdout);

        // advance forward
        token = next;
        pos++;
    }

    // report achieved tok/s
    long end = time_in_ms();
    printf("\nachieved tok/s: %f\n", steps / (double)(end-start)*1000);

    // memory and file handles cleanup
    free_run_state(&state);
    for (int i = 0; i < config.vocab_size; i++) { free(vocab[i]); }
    free(vocab);
    if (data != MAP_FAILED) munmap(data, file_size);
    if (fd != -1) close(fd);
    return 0;
}
