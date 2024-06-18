import yaml
import os
import numpy as np
import copy
import tqdm
from concurrent.futures import ThreadPoolExecutor
import re
import random
import json
import ollama
from openai import OpenAI
import requests



def load_config_from_yaml(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = yaml.safe_load(file)
    return data



class LLM_Base():

    def __init__(self,
                 max_retries=3,
                 beta = 3,
                 show_prompt = None,
                 show_response = None,
                 show_progress = None,
                 parallel=None,
                 llm_name=None,
                 config_path = None,
                 num_processes = None,
                 backend = None,
                 show_token_size = False
                 ) -> None:
        


        self.fit_prompts = None

        if config_path is None:
            project_root = os.path.abspath(os.path.dirname(__file__))
            config_path = os.path.join(project_root, 'config.yaml')
        self.config = load_config_from_yaml(config_path)
        self.infer_config = self.config.get('infer_config')


        self.max_retries = self.infer_config.get('max_retries', 3) if max_retries is None else max_retries
        self.beta = self.infer_config.get('beta', 3) if beta is None else beta
        self.show_prompt = self.infer_config.get('show_prompt', False) if show_prompt is None else show_prompt
        self.show_response = self.infer_config.get('show_response', False) if show_response is None else show_response
        self.show_progress = self.infer_config.get('show_progress', True) if show_progress is None else show_progress
        self.parallel = self.infer_config.get('parallel', False) if parallel is None else parallel
        self.num_processes = self.infer_config.get('num_processes', 1) if num_processes is None else num_processes
        self.backend = self.infer_config.get('backend', None) if backend is None else backend
        self.show_token_size = self.infer_config.get('show_token_size', False) if show_token_size is None else show_token_size



        if self.backend == 'ollama':
            if llm_name is None:
                self.llm_name = self.config.get('ollama').get('model')  
            else:
                self.llm_name = llm_name
            
            ollama_models = ollama.Client(host=self.config.get('ollama').get('url')).list()['models']
            names = [model['name'] for model in ollama_models]
            if self.llm_name not in names:
                raise Exception(f"{self.llm_name} is not in the Ollama model list.")
            

        elif self.backend == 'vllm':
            if llm_name is None:
                self.llm_name = self.config.get('vllm').get('model')
            else:
                self.llm_name = llm_name

        elif self.backend == 'openai':
            if llm_name is None:
                self.llm_name = self.config.get('openai').get('model')
            else:
                self.llm_name = llm_name

        else:
            raise Exception(f"backend {self.backend} is not supported.")





    def fit(self,Xs,ys):
        """
        Store the training data for the model
        """

        # Convert Xs and ys to numpy arrays
        if not isinstance(Xs, np.ndarray):
            Xs = np.array(Xs)
        if Xs.ndim < 1:
            Xs = Xs.reshape(1, -1)
        elif Xs.ndim == 1:
            Xs = Xs.reshape(1, -1)

        ys = np.array(ys).flatten()

        # Store the training data
        self.Train_Xs = Xs
        self.Train_ys = ys


    def normalize(self, Train_Xs, Test_Xs, range_01=True):
        """
        Normalize the training and testing data.
        """
        combined_Xs = np.vstack((Train_Xs, Test_Xs))
        max_Xs = np.max(combined_Xs, axis=0)
        min_Xs = np.min(combined_Xs, axis=0)

        if range_01:
            Norm_Train_Xs = (Train_Xs - min_Xs) / (max_Xs - min_Xs)
            Norm_Test_Xs = (Test_Xs - min_Xs) / (max_Xs - min_Xs)
        else:
            Norm_Train_Xs = 2 * ((Train_Xs - min_Xs) / (max_Xs - min_Xs)) - 1
            Norm_Test_Xs = 2 * ((Test_Xs - min_Xs) / (max_Xs - min_Xs)) - 1

        Norm_Train_Xs = np.round(Norm_Train_Xs, self.beta)
        Norm_Test_Xs = np.round(Norm_Test_Xs, self.beta)

        return Norm_Train_Xs, Norm_Test_Xs




    def predict(self, Test_Xs):  

        if self.Train_Xs is None or self.Train_ys is None:
            raise Exception("Train_Xs and Train_ys are None. Please call fit() first.")
        
        if not isinstance(Test_Xs, np.ndarray):
            Test_Xs = np.array(Test_Xs)
        if Test_Xs.ndim < 1:
            Test_Xs = Test_Xs.reshape(1, -1)
        elif Test_Xs.ndim == 1:
            Test_Xs = Test_Xs.reshape(1, -1)


        # Normalize the data
        Norm_Train_Xs, Norm_Test_Xs = self.normalize(self.Train_Xs, Test_Xs)

        if self.__class__.__name__ == 'LLM_Regression':
            # get predict prompts
            Train_ys = self.normalize_ys(self.Train_ys)
        else:
            Train_ys = self.Train_ys

        # get fit prompts
        self.fit_prompts = self.generate_fit_prompts(Norm_Train_Xs, Train_ys)



        if self.parallel:
            # parallel
            return self.predict_parallel(Norm_Test_Xs)
        else:
            # serial
            return self.predict_serial(Norm_Test_Xs)
        

    def predict_serial(self, Norm_Test_Xs):
        res = []

        for X in tqdm.tqdm(Norm_Test_Xs, disable=not self.show_progress):
            prompts = self.generate_predict_prompts(X)
            final_prompt = " ".join(prompts)

            if self.show_prompt:
                print(final_prompt)  # print the prompt

            y_pred = self.call_llm(final_prompt)
            res.append(y_pred)


        if self.__class__.__name__ == 'LLM_Regression':
            res = np.array(res) * (self.ys_max - self.ys_min) + self.ys_min
        return np.array(res).flatten()


    def predict_parallel(self, Norm_Test_Xs):

        def process(X):
            prompts = self.generate_predict_prompts(X)
            final_prompt = " ".join(prompts)

            y_pred = self.call_llm(final_prompt)
            return y_pred
        
        with ThreadPoolExecutor(max_workers=self.num_processes) as executor:
            res = list(tqdm.tqdm(executor.map(process, Norm_Test_Xs), total=len(Norm_Test_Xs), disable=not self.show_progress))
        


        if self.__class__.__name__ == 'LLM_Regression':
            res = np.array(res) * (self.ys_max - self.ys_min) + self.ys_min
        return np.array(res).flatten()
    



    def send_request_ollama(self, prompts,timeout=100):


        endpoint = '/api/generate'
        url = self.config.get('ollama').get('url')+endpoint

        data = {
            "model": self.llm_name,
            "prompt": prompts,
            "format": "json",
            "stream": False
        }
        headers = {"Content-Type": "application/json"}
        try:
            response = requests.post(url, data=json.dumps(data), headers=headers, timeout=timeout)
            response.raise_for_status()  
            return response.json()['response']
            # return response.text
        except requests.exceptions.Timeout:
            return "Request timed out. Please check your network connection or adjust the timeout settings."
        except requests.exceptions.HTTPError as err:
            return "HTTP error: {}".format(err)
        except requests.exceptions.RequestException as e:
            return "Request exception: {}".format(e)




    def send_request_openai(self, prompts):

        endpoint = '/v1'


        client = OpenAI(
            api_key=self.config.get(self.backend).get('api_key'),
            base_url = self.config.get(self.backend).get('url')+endpoint
        )

        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompts,
                },
            ],
            model=self.llm_name,
            max_tokens=10,
        )
        res = chat_completion.choices[0].message.content
        return res



    def call_llm(self, prompt):
        """
        Call the language model to predict the data
        """
        raise NotImplementedError("call_llm method not implemented")


    def generate_fit_prompts(self, Xs, ys):
        """
        Generate the prompts for the model to fit the data
        """
        raise NotImplementedError("generate_fit_prompts method not implemented")
    

    def generate_predict_prompts(self, X):
        """
        Generate the prompts for the model to predict the data
        """
        raise NotImplementedError("generate_predict_prompts method not implemented")







class LLM_Classification(LLM_Base):

    def generate_fit_prompts(self, Xs, ys):
        prompts = [self.config.get('prompt_template').get('cla_introduction')]

        for row,label in zip(Xs,ys):
            if label == 1:
                p = f"Features: <{', '.join(map(str, row))}>, Class: better\n"
            else:
                p = f"Features: <{', '.join(map(str, row))}>, Class: worse\n"
            prompts.append(p)

        return prompts
    
    def generate_predict_prompts(self, X):
        if self.fit_prompts is None:
            raise Exception("fit_prompts is None. Please call generate_fit_prompts() first.")
        
        prompts = copy.deepcopy(self.fit_prompts)
        
        prompts.append( "\n\nNew Evaluation:\n")

        p = f"<{', '.join(map(str, X))}>  better or worse?"
        prompts.append(p)
        prompts.append("\n\nNote: Respond in Json with the format {'Class':'result'} only.")

        return prompts


    def call_llm(self, prompts):
        """
        Call the language model to predict the data
        """
        if self.backend is None:
            raise Exception("backend is None. Please set the backend.")
        for _ in range(self.max_retries):
            try:
                if self.backend == 'ollama':
                    raw_res = self.send_request_ollama(prompts=prompts)
                if self.backend == 'vllm':
                    raw_res = self.send_request_openai(prompts=prompts)
                if self.backend == 'openai':
                    raw_res = self.send_request_openai(prompts=prompts)
            except:
                Warning("Error in request")
                continue

            if self.show_token_size:
                pass

            json_str = raw_res.strip()
            json_str = json_str.replace("'", '"')
            json_str = re.sub(r'\s+', ' ', json_str)
            

            if self.show_response:
                print(json_str)

            match = re.search(r'\{.*?\}', json_str)
            if match:
                json_part = match.group(0)
                json_part = json_part.replace("'", '"')

                try:
                    data = json.loads(json_part)
                    if data['Class'].lower() == 'better':
                        return 1
                    elif data['Class'].lower() == 'worse':
                        return -1
                except:
                    continue


        return random.choice([-1, 1])
            

class LLM_Regression(LLM_Base):

    def __init__(self, max_retries=3, beta=3, show_prompt=None, show_response=None, show_progress=None, parallel=None, llm_name=None, config_path=None) -> None:
        super().__init__(max_retries, beta, show_prompt, show_response, show_progress, parallel, llm_name, config_path)


        self.ys_max = None
        self.ys_min = None

    def normalize_ys(self, ys):
        """
        normalize the target values
        """
        self.ys_max = np.max(ys)
        self.ys_min = np.min(ys)

        return (ys - self.ys_min) / (self.ys_max - self.ys_min)



    def generate_fit_prompts(self, Xs, ys):
        prompts = [self.config.get('prompt_template').get('reg_introduction')]

        for row, value in zip(Xs,ys):
            value = value.flatten().tolist()[0]
            value = round(value, 5)

            p = f"Features: <{', '.join(map(str, row))}> Value: {value}\n"
            prompts.append(p)

        return prompts
    
    def generate_predict_prompts(self, X):
        if self.fit_prompts is None:
            raise Exception("fit_prompts is None. Please call generate_fit_prompts() first.")
        
        prompts = copy.deepcopy(self.fit_prompts)
        
        prompts.append( "\n\nNew Evaluation:\n")

        p = f"<{', '.join(map(str, X))}>  Target?"
        prompts.append(p)
        prompts.append("\n\nNote: Respond in Json with the format {'Target':'result'} only.")

        return prompts



    def call_llm(self, prompts):
        """
        Call the language model to predict the data
        """
        if self.backend is None:
            raise Exception("backend is None. Please set the backend.")
        
        for _ in range(self.max_retries):
            try:
                if self.backend == 'ollama':
                    raw_res = self.send_request_ollama(prompts=prompts)
                if self.backend == 'vllm':
                    raw_res = self.send_request_openai(prompts=prompts)
                if self.backend == 'openai':
                    raw_res = self.send_request_openai(prompts=prompts)
            except:
                Warning("Error in request")
                continue
    

            if self.show_token_size:
                pass

            json_str = raw_res.strip()
            json_str = json_str.replace("'", '"')
            json_str = re.sub(r'\s+', ' ', json_str)
            

            if self.show_response:
                print(json_str)

            match = re.search(r'\{.*?\}', raw_res, re.DOTALL)
            if match:
                json_part = match.group(0)
                json_part = json_part.replace("'", '"')

                try:
                    data = json.loads(json_part)
                    value = float(data['Value'])  #
                    return value  
                except (json.JSONDecodeError, ValueError, KeyError) as e:

                    continue

        return random.random() 




if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split


    n_features = 5

    # For Classification
    X, y = make_classification(n_samples=100, n_features=n_features,random_state=42)
    y = y * 2 - 1
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)


    model = LLM_Classification()

    model.fit(X_train, y_train)
    pre = model.predict(X_test[:2])
    print(pre)


    # For Regression
    from sklearn.datasets import make_regression
    X, y = make_regression(n_samples=100, n_features=n_features,random_state=42)
    y.flatten()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
    model = LLM_Regression()

    model.fit(X_train, y_train)
    pre = model.predict(X_test[:2])
    print(pre)