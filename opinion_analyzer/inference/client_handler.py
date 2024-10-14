""" ClientHandler class for text generation using vLLM (offline) or transformers pipeline. """

import traceback
import argparse
import transformers
import torch
from transformers import AutoModelForCausalLM, AutoModel, AutoTokenizer
from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from PIL import Image

from opinion_analyzer.data_handler.prompt_templates import template_dict
from opinion_analyzer.data_handler.prompt_database import is_argument
from opinion_analyzer.utils.helper import (
    get_model_client_config,
    make_sentences_concrete
)
from opinion_analyzer.utils.log import get_logger

model_client_config = get_model_client_config()

log = get_logger()


class ClientHandler:  # pylint: disable=too-many-instance-attributes
    """Uses either the vLLM API or the transformers pipeline to generate text based on a prompt."""

    def __init__(self, model_name_or_path: str = None, tokenizer_model: str = None):
        # pylint: disable=too-many-arguments
        self.processor = None
        self.model = None
        self.model_config = None
        self.bnb_config = None
        self.tokenizer = None
        self.model_name_or_path = model_name_or_path
        if self.model_name_or_path is not None:
            if tokenizer_model is None:
                self.tokenizer_model = model_name_or_path
            else:
                self.tokenizer_model = tokenizer_model
            self.pipeline = self.get_text_gen_pipeline()
            self.model_client = model_client_config[self.model_name_or_path]["client"]

    def find_config(self):
        """
        Finds the right config for transformers' pipeline.
        """
        if "mistral" in self.model_name_or_path.lower():
            return transformers.MistralConfig
        return transformers.AutoConfig

    def get_transformers_pipeline(self):
        """
        Prepares the transformers text generation pipeline.
        """

        self.bnb_config = transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            llm_int8_enable_fp32_cpu_offload=True,
        )

        config_class = self.find_config()
        self.model_config = config_class.from_pretrained(
            self.model_name_or_path
        )  # use_auth_token=hf_auth

        # initialize the model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            trust_remote_code=False,  # True for flash-attn2 else False
            config=self.model_config,
            device_map="auto",
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_model,
        )

        text_gen_pipeline = transformers.pipeline(
            model=self.model,
            tokenizer=self.tokenizer,
            return_full_text=False,  # langchain expects the full text
            task="text-generation",
        )
        return text_gen_pipeline

    def get_vi_pipeline_1(self):
        """
        Loads the model and the tokenizer for visual instruction (vi) model,
        such as openbmb/MiniCPM-Llama3-V-2_5.

        :return: Nothing
        """
        model = AutoModel.from_pretrained(
            self.model_name_or_path, trust_remote_code=True, torch_dtype=torch.float16
        )
        model = model.to(device="cuda")
        model.eval()
        self.model = model

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path, trust_remote_code=True
        )

    def get_vi_pipeline_2(self):
        """
        Loads the model and the tokenizer for a visual instruction (vi) model,
        such as microsoft/Phi-3-vision-128k-instruct.

        :return: None
        """

        model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            device_map="cuda",
            trust_remote_code=True,
            torch_dtype="auto",
            # _attn_implementation="flash_attention_2",
            # use _attn_implementation='eager' to disable flash attention
            _attn_implementation="eager",
        )
        self.model = model
        processor = AutoProcessor.from_pretrained(
            self.model_name_or_path, trust_remote_code=True
        )

        self.processor = processor

    def get_text_gen_pipeline(self):
        """
        Finds the adequate text generation pipeline for your model.

        :return: Text generation pipeline or None.
        """
        if model_client_config[self.model_name_or_path]["client"] == "transformers":
            return self.get_transformers_pipeline()

        if model_client_config[self.model_name_or_path]["client"] == "vi":
            if self.model_name_or_path == "openbmb/MiniCPM-Llama3-V-2_5":
                return self.get_vi_pipeline_1()
            if self.model_name_or_path == "microsoft/Phi-3-vision-128k-instruct":
                return self.get_vi_pipeline_2()

        output = None
        try:
            output = LLM(
                self.model_name_or_path, max_model_len=512 * 4, trust_remote_code=True
            )
        except Exception as e:  # pylint: disable=broad-exception-caught
            print(e)
            log.error(e)
            log.error("Traceback: %s", traceback.format_exc())

        return output

    def rename_params(self, params: dict) -> dict:
        """
        Renames certain hyperparameters so that they can be read by the active text generation pipeline.

        :param: Hyperparameters as dictionary to teak text generation.
        """

        if model_client_config[self.model_name_or_path]["client"] in [
            "transformers",
            "vi",
        ]:
            if "max_tokens" in params.keys():
                params["max_new_tokens"] = params["max_tokens"]
                del params["max_tokens"]
        return params

    def generate_vi(
            self, prompt: str, params: dict = None, img: Image.Image = None
    ) -> str:
        """
        Generate a response from a visual instruction model based on a given prompt and an optional image.

        Depending on the `model_name_or_path` attribute of the class instance, this function utilizes different
        text generation pipelines to generate a response.

        - For "VAGOsolutions/Llama-3-SauerkrautLM-70b-Instruct", an image must be provided.
        - For "microsoft/Phi-3-vision-128k-instruct", a prompt and optional image are processed to generate text.

        :param prompt:
            The textual prompt for the model to generate a response from.
        :type prompt: str

        :param params:
            Optional parameters for the text generation process, such as decoding strategies.
            If not provided, default parameters are used.
        :type params: dict, optional

        :param img:
            An optional image input for visual-instruction models that require an image to generate text.
        :type img: Image.Image, optional

        :return:
            A generated text response based on the input prompt and, if applicable, the provided image.
        :rtype: str

        :raises ValueError:
            If an image is not provided for models that require one or if no valid text generation pipeline is found.
        """
        if img is None:
            raise ValueError("You need to pass an image.")

        if self.model_name_or_path == "openbmb/MiniCPM-Llama3-V-2_5":
            messages = [{"role": "user", "content": prompt}]

            res = self.model.chat(
                image=img,
                msgs=messages,
                tokenizer=self.tokenizer,
                sampling=True,  # if sampling=False, beam_search will be used by default
                temperature=0.7,
            )

            return res

        if self.model_name_or_path == "microsoft/Phi-3-vision-128k-instruct":
            messages = [{"role": "user", "content": f"<|image_1|>\n{prompt}"}]

            prompt = self.processor.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            inputs = self.processor(prompt, [img], return_tensors="pt").to(device)

            generate_ids = self.model.generate(
                **inputs, eos_token_id=self.processor.tokenizer.eos_token_id, **params
            )

            # Slice the generated token IDs to exclude the input tokens
            generate_ids = generate_ids[:, inputs["input_ids"].shape[1]:]

            response = self.processor.batch_decode(
                generate_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]

            return response

        raise ValueError("No valid text generation pipeline found.")

    def generate(
            self, prompt: str, params: dict = None, img: Image.Image = None
    ) -> str:
        """
        Handles the text generation based on the active text generation pipeline.

        :param prompt: Prompt for the LLM.
        :param params: Hyperparameters as dictionary to tweak text generation.
        :param img: In case you are using a visual question answering model, paste an image.
        """
        params = params or model_client_config["default_hyperparams"]
        params = self.rename_params(params)

        if model_client_config[self.model_name_or_path]["template"] is not None:
            prompt = template_dict[
                model_client_config[self.model_name_or_path]["template"]
            ].format(prompt=prompt)

        if isinstance(self.pipeline, LLM):
            sampling_params = SamplingParams(**params)
            outputs = self.pipeline.generate([prompt], sampling_params)
            return outputs[0].outputs[0].text

        if model_client_config[self.model_name_or_path]["client"] == "transformers":
            return self.pipeline(prompt, **params)[0]["generated_text"]

        if model_client_config[self.model_name_or_path]["client"] == "vi":
            return self.generate_vi(prompt=prompt, params=params, img=img)

        raise ValueError("No valid text generation pipeline found.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-mnp",
        "--model_name_or_path",
        default="VAGOsolutions/Llama-3-SauerkrautLM-70b-Instruct",
        type=str,
        help="Specify the model name.",
    )

    parser.add_argument(
        "-pti",
        "--path_to_image",
        type=str,
        default=None,
        help="Specify the path to the image you want to analyze.",
    )

    args = parser.parse_args()
    ch = ClientHandler(args.model_name_or_path)

    context = """Am 23. Januar 2020 reichte die Sozialdemokratische Partei der Schweiz die Volksinitiative «Maximal 10 % des Einkommens für die Krankenkassenprämien (Prämien-Entlastungs-Initiative)» mit rund 118’00 Unterschriften ein. Die Initiative will, dass Bund und Kantone die Versicherten bei den Prämien für die obligatorische Krankenpflegeversicherung entlasten. 

Der Bundesrat beantragt, die Initiative abzulehnen und – als indirekten Gegenvorschlag – mit einer Änderung des Bundesgesetzes über die Krankenversicherung die Kantone zu verpflichten, die Prämienverbilligung so zu regeln, dass diese jährlich einem Mindestanteil der Bruttokosten der obligatorischen Krankenpflegeversicherung entspricht. 

Die Prämien-Entlastungs-Initiative fordert, dass die Versicherten höchstens 10 Prozent ihres verfügbaren Einkommens für die Prämien aufwenden müssen. Die Prämienverbilligung soll zu mindestens zwei Drittel durch den Bund und zum verbleibenden Betrag durch die Kantone finanziert werden. 

Heute verpflichtet das Bundesgesetz über die Krankenversicherung die Kantone, die Prämien der Versicherten in bescheidenen wirtschaftlichen Verhältnissen zu verbilligen. Der Bund gewährt den Kantonen dazu einen Beitrag. Im Jahr 2020 haben der Bund die Prämien mit 2,9 Milliarden Franken und die Kantone mit 2,6 Milliarden Franken verbilligt. 

Der Bundesrat kann das Anliegen der Initiative, die Versicherten in bescheidenen wirtschaftlichen Verhältnissen zu entlasten, nachvollziehen. Er erachtet es als problematisch, dass mehrere Kantone ihren Anteil an der Finanzierung der Prämienverbilligung in den letzten Jahren gesenkt haben. 

Der Bund müsste jedoch bedeutend mehr Mittel zur Verfügung stellen als bisher, da er mindestens zwei Drittel der Prämienverbilligung finanzieren müsste. Damit müsste er auch für Kosten aufkommen, die von den Kantonen beeinflusst werden können. Die Kantone beeinflussen über ihre Spitalplanung die Spitalkosten und über die Steuerung der Zulassung von Leistungserbringern die ambulanten Kosten. Weil die Prämien zudem stärker steigen als die Einkommen, führt die Initiative rasch zu hohen Mehrkosten für Bund und Kantone. Die Initiative konzentriert sich weiter ausschliesslich auf die Finanzierung. Die Kosten müssen jedoch auch gedämpft werden. 

Der Bundesrat beantragt, die Prämien-Entlastungs-Initiative abzulehnen und einer Änderung des Bundesgesetzes über die Krankenversicherung als indirektem Gegenvorschlag zuzustimmen. Die Kantone sollen verpflichtet werden, die Prämienverbilligung so zu regeln, dass sie einem Mindestanteil der Bruttokosten der obligatorischen Krankenpflegeversicherung im betreffenden Kanton entspricht. Damit erhalten die Kantone einen Anreiz, ihre Bruttokosten zu dämpfen. Die Höhe des Anteils soll davon abhängen, wie stark die verbilligten Prämien die Versicherten mit den untersten Einkommen im betreffenden Kanton belasten. Die Kantone sollen weiterhin bestimmen können, wie sie die Prämienverbilligung ausgestalten.

Gemäss Schätzungen des Bundesamtes für Gesundheit (BAG) in der Botschaft belaufen sich die Mehrkosten zulasten der Kantone mit dem Gegenvorschlag im Jahr 2024 rund 600 Millionen Franken. Bei Annahme der Initiative würden sich die Mehrkosten zulasten der Kantone auf 1,1 Milliarde Franken belaufen."""

    sentence = "Am 23. Januar 2020 reichte die Sozialdemokratische Partei der Schweiz die Volksinitiative «Maximal 10 % des Einkommens für die Krankenkassenprämien (Prämien-Entlastungs-Initiative)» mit rund 118’00 Unterschriften ein."

    prompt = is_argument.format(topic_text="Kindergeld sollte verboten werden.",
                                text_sample="Kindergeld ist nicht wichtig, da es vor allem Frauen nicht unterstützt, weiterhin berufstätig zu sein.")

    answer = ch.generate(
        prompt
    )
    print(answer)

    # results = make_sentences_concrete(text=context, client_handler=ch)
    # results.to_excel("sentence_rewriting.xlsx", index=False)
