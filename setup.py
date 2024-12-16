from setuptools import setup, find_packages

setup(
    name='aitutor_assessmentkit',  # Package name
    version='0.1.8',  # Package version
    packages=find_packages(),  # Automatically discover and include packages
    install_requires=[  # List of dependencies required for the package
        "numpy",  # Numerical computations
        "torch",  # PyTorch for deep learning tasks
        "pandas",  # Data manipulation
        "tqdm",  # Progress bars
        "scipy",  # Scientific computing
        "nltk",  # Natural language processing
        "transformers",  # Transformer models
        "scikit-learn",  # Machine learning library
        "matplotlib",  # Plotting and visualization
        "seaborn",  # Statistical data visualization
        "bert-score",  # BERT-based evaluation metric
        "clean-text",  # Text cleaning utilities
        "datasets",  # Hugging Face datasets library
        "evaluate",  # Evaluation utilities
        "huggingface-hub",  # Hugging Face model hub integration
        "llvmlite",  # LLVM bindings
        "notebook",  # Jupyter Notebook support
        "openai",  # OpenAI API integration
        "sentencepiece",  # Tokenizer library for text processing
        "stanza",  # NLP library
        "tokenizers",  # Fast tokenizer library
        "num2words",
        "vllm",
    ],
    long_description=open('README.md').read(),  # Long description from README.md
    long_description_content_type='text/markdown',  # Format of the long description
    author='Kaushal Kumar Maurya',  # Author name
    author_email='Kaushal.Maurya@mbzuai.ac.ae',  # Author email
    description=(
        'AITutor-AssessmentKit is the first open-source toolkit designed to evaluate the '
        'pedagogical performance of AI tutors in student mistake remediation tasks. With the '
        'growing capabilities of large language models (LLMs), this library provides a systematic '
        'approach to assess their teaching potential across multiple dimensions in educational dialogues.'
    ),  # Short description of the package
    url='https://github.com/kaushal0494/aitutor-assessmentkit',  # Package repository URL
    classifiers=[  # PyPI classifiers to categorize the package
        "Programming Language :: Python :: 3.10",
        'License :: OSI Approved :: MIT License',  # License type
        'Operating System :: OS Independent',  # Compatibility with operating systems
    ],
    python_requires='>=3.10',  # Minimum Python version requirement
)