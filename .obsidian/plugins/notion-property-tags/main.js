// Notion Property Tags — generated for this vault.
// Directly styles tag pills inside page properties, independent of the Colored Tags plugin selectors.
const { Plugin } = require('obsidian');

const COLOR_MAP = {
  "recommended": {
    "light": {
      "bg": "#FBF3DB",
      "fg": "#CB912F"
    },
    "dark": {
      "bg": "#59563B",
      "fg": "#FFDC49"
    },
    "name": "yellow"
  },
  "to-read": {
    "light": {
      "bg": "#F1F1EF",
      "fg": "#37352F"
    },
    "dark": {
      "bg": "#2F2F2C",
      "fg": "#D4D4D4"
    },
    "name": "default"
  },
  "multimodal": {
    "light": {
      "bg": "#E7F3F8",
      "fg": "#337EA9"
    },
    "dark": {
      "bg": "#28456C",
      "fg": "#529CCA"
    },
    "name": "blue"
  },
  "domain-generalization": {
    "light": {
      "bg": "#FDEBEC",
      "fg": "#D44C47"
    },
    "dark": {
      "bg": "#594141",
      "fg": "#FF7369"
    },
    "name": "red"
  },
  "concept": {
    "light": {
      "bg": "#E7F3F8",
      "fg": "#337EA9"
    },
    "dark": {
      "bg": "#28456C",
      "fg": "#529CCA"
    },
    "name": "blue"
  },
  "to-review": {
    "light": {
      "bg": "#FBF3DB",
      "fg": "#CB912F"
    },
    "dark": {
      "bg": "#59563B",
      "fg": "#FFDC49"
    },
    "name": "yellow"
  },
  "self-contact-hand-face-cv": {
    "light": {
      "bg": "#EDF3EC",
      "fg": "#448361"
    },
    "dark": {
      "bg": "#354C4B",
      "fg": "#4DAB9A"
    },
    "name": "green"
  },
  "lesson-summary": {
    "light": {
      "bg": "#E7F3F8",
      "fg": "#337EA9"
    },
    "dark": {
      "bg": "#28456C",
      "fg": "#529CCA"
    },
    "name": "blue"
  },
  "not-started": {
    "light": {
      "bg": "#F1F1EF",
      "fg": "#37352F"
    },
    "dark": {
      "bg": "#2F2F2C",
      "fg": "#D4D4D4"
    },
    "name": "default"
  },
  "done": {
    "light": {
      "bg": "#EDF3EC",
      "fg": "#448361"
    },
    "dark": {
      "bg": "#354C4B",
      "fg": "#4DAB9A"
    },
    "name": "green"
  },
  "cognition": {
    "light": {
      "bg": "#FBF3DB",
      "fg": "#CB912F"
    },
    "dark": {
      "bg": "#59563B",
      "fg": "#FFDC49"
    },
    "name": "yellow"
  },
  "dataset": {
    "light": {
      "bg": "#EDF3EC",
      "fg": "#448361"
    },
    "dark": {
      "bg": "#354C4B",
      "fg": "#4DAB9A"
    },
    "name": "green"
  },
  "theory": {
    "light": {
      "bg": "#F6F3F9",
      "fg": "#9065B0"
    },
    "dark": {
      "bg": "#44355B",
      "fg": "#9A6DD7"
    },
    "name": "purple"
  },
  "contact-aware": {
    "light": {
      "bg": "#EDF3EC",
      "fg": "#448361"
    },
    "dark": {
      "bg": "#354C4B",
      "fg": "#4DAB9A"
    },
    "name": "green"
  },
  "paradigms": {
    "light": {
      "bg": "#F4EEEE",
      "fg": "#9F6B53"
    },
    "dark": {
      "bg": "#493A34",
      "fg": "#D4A181"
    },
    "name": "brown"
  },
  "hand-face-cv": {
    "light": {
      "bg": "#EDF3EC",
      "fg": "#448361"
    },
    "dark": {
      "bg": "#354C4B",
      "fg": "#4DAB9A"
    },
    "name": "green"
  },
  "in-progress": {
    "light": {
      "bg": "#E7F3F8",
      "fg": "#337EA9"
    },
    "dark": {
      "bg": "#28456C",
      "fg": "#529CCA"
    },
    "name": "blue"
  },
  "turn-taking": {
    "light": {
      "bg": "#FBECDD",
      "fg": "#D9730D"
    },
    "dark": {
      "bg": "#594A36",
      "fg": "#FFA344"
    },
    "name": "orange"
  },
  "cv": {
    "light": {
      "bg": "#EDF3EC",
      "fg": "#448361"
    },
    "dark": {
      "bg": "#354C4B",
      "fg": "#4DAB9A"
    },
    "name": "green"
  },
  "dataset-corpus": {
    "light": {
      "bg": "#EDF3EC",
      "fg": "#448361"
    },
    "dark": {
      "bg": "#354C4B",
      "fg": "#4DAB9A"
    },
    "name": "green"
  },
  "essential": {
    "light": {
      "bg": "#FDEBEC",
      "fg": "#D44C47"
    },
    "dark": {
      "bg": "#594141",
      "fg": "#FF7369"
    },
    "name": "red"
  },
  "multimodal-disfluency-and-discourse": {
    "light": {
      "bg": "#E7F3F8",
      "fg": "#337EA9"
    },
    "dark": {
      "bg": "#28456C",
      "fg": "#529CCA"
    },
    "name": "blue"
  },
  "background": {
    "light": {
      "bg": "#F1F1EF",
      "fg": "#37352F"
    },
    "dark": {
      "bg": "#2F2F2C",
      "fg": "#D4D4D4"
    },
    "name": "default"
  },
  "dynamic-programming": {
    "light": {
      "bg": "#E7F3F8",
      "fg": "#337EA9"
    },
    "dark": {
      "bg": "#28456C",
      "fg": "#529CCA"
    },
    "name": "blue"
  },
  "gesture-and-turn-taking": {
    "light": {
      "bg": "#FBECDD",
      "fg": "#D9730D"
    },
    "dark": {
      "bg": "#594A36",
      "fg": "#FFA344"
    },
    "name": "orange"
  },
  "benchmark": {
    "light": {
      "bg": "#F4EEEE",
      "fg": "#9F6B53"
    },
    "dark": {
      "bg": "#493A34",
      "fg": "#D4A181"
    },
    "name": "brown"
  },
  "cvpr": {
    "light": {
      "bg": "#E7F3F8",
      "fg": "#337EA9"
    },
    "dark": {
      "bg": "#28456C",
      "fg": "#529CCA"
    },
    "name": "blue"
  },
  "leetcode": {
    "light": {
      "bg": "#F6F3F9",
      "fg": "#9065B0"
    },
    "dark": {
      "bg": "#44355B",
      "fg": "#9A6DD7"
    },
    "name": "purple"
  },
  "medium": {
    "light": {
      "bg": "#E7F3F8",
      "fg": "#337EA9"
    },
    "dark": {
      "bg": "#28456C",
      "fg": "#529CCA"
    },
    "name": "blue"
  },
  "problem": {
    "light": {
      "bg": "#EDEDEB",
      "fg": "#6B6B67"
    },
    "dark": {
      "bg": "#373737",
      "fg": "#B7B7B5"
    },
    "name": "gray"
  },
  "reading": {
    "light": {
      "bg": "#E7F3F8",
      "fg": "#337EA9"
    },
    "dark": {
      "bg": "#28456C",
      "fg": "#529CCA"
    },
    "name": "blue"
  },
  "to-solve": {
    "light": {
      "bg": "#FBECDD",
      "fg": "#D9730D"
    },
    "dark": {
      "bg": "#594A36",
      "fg": "#FFA344"
    },
    "name": "orange"
  },
  "arxiv": {
    "light": {
      "bg": "#EDEDEB",
      "fg": "#6B6B67"
    },
    "dark": {
      "bg": "#373737",
      "fg": "#B7B7B5"
    },
    "name": "gray"
  },
  "disfluency": {
    "light": {
      "bg": "#F9EEF3",
      "fg": "#C14C8A"
    },
    "dark": {
      "bg": "#533B4C",
      "fg": "#E255A1"
    },
    "name": "pink"
  },
  "gpu": {
    "light": {
      "bg": "#FBECDD",
      "fg": "#D9730D"
    },
    "dark": {
      "bg": "#594A36",
      "fg": "#FFA344"
    },
    "name": "orange"
  },
  "none": {
    "light": {
      "bg": "#EDEDEB",
      "fg": "#6B6B67"
    },
    "dark": {
      "bg": "#373737",
      "fg": "#B7B7B5"
    },
    "name": "gray"
  },
  "skim-skip": {
    "light": {
      "bg": "#FBECDD",
      "fg": "#D9730D"
    },
    "dark": {
      "bg": "#594A36",
      "fg": "#FFA344"
    },
    "name": "orange"
  },
  "ml": {
    "light": {
      "bg": "#E7F3F8",
      "fg": "#337EA9"
    },
    "dark": {
      "bg": "#28456C",
      "fg": "#529CCA"
    },
    "name": "blue"
  },
  "not-relevant": {
    "light": {
      "bg": "#FDEBEC",
      "fg": "#D44C47"
    },
    "dark": {
      "bg": "#594141",
      "fg": "#FF7369"
    },
    "name": "red"
  },
  "action-recognition-backbone": {
    "light": {
      "bg": "#EDEDEB",
      "fg": "#6B6B67"
    },
    "dark": {
      "bg": "#373737",
      "fg": "#B7B7B5"
    },
    "name": "gray"
  },
  "hardware": {
    "light": {
      "bg": "#F4EEEE",
      "fg": "#9F6B53"
    },
    "dark": {
      "bg": "#493A34",
      "fg": "#D4A181"
    },
    "name": "brown"
  },
  "llm": {
    "light": {
      "bg": "#F6F3F9",
      "fg": "#9065B0"
    },
    "dark": {
      "bg": "#44355B",
      "fg": "#9A6DD7"
    },
    "name": "purple"
  },
  "data-structures": {
    "light": {
      "bg": "#F6F3F9",
      "fg": "#9065B0"
    },
    "dark": {
      "bg": "#44355B",
      "fg": "#9A6DD7"
    },
    "name": "purple"
  },
  "journal": {
    "light": {
      "bg": "#EDEDEB",
      "fg": "#6B6B67"
    },
    "dark": {
      "bg": "#373737",
      "fg": "#B7B7B5"
    },
    "name": "gray"
  },
  "lesson-day": {
    "light": {
      "bg": "#EDEDEB",
      "fg": "#6B6B67"
    },
    "dark": {
      "bg": "#373737",
      "fg": "#B7B7B5"
    },
    "name": "gray"
  },
  "read": {
    "light": {
      "bg": "#EDF3EC",
      "fg": "#448361"
    },
    "dark": {
      "bg": "#354C4B",
      "fg": "#4DAB9A"
    },
    "name": "green"
  },
  "adaptor-and-self-touch-theory": {
    "light": {
      "bg": "#F6F3F9",
      "fg": "#9065B0"
    },
    "dark": {
      "bg": "#44355B",
      "fg": "#9A6DD7"
    },
    "name": "purple"
  },
  "pose-estimation": {
    "light": {
      "bg": "#EDF3EC",
      "fg": "#448361"
    },
    "dark": {
      "bg": "#354C4B",
      "fg": "#4DAB9A"
    },
    "name": "green"
  },
  "self-adaptor": {
    "light": {
      "bg": "#EDF3EC",
      "fg": "#448361"
    },
    "dark": {
      "bg": "#354C4B",
      "fg": "#4DAB9A"
    },
    "name": "green"
  },
  "weakly": {
    "light": {
      "bg": "#E7F3F8",
      "fg": "#337EA9"
    },
    "dark": {
      "bg": "#28456C",
      "fg": "#529CCA"
    },
    "name": "blue"
  },
  "iccv": {
    "light": {
      "bg": "#E7F3F8",
      "fg": "#337EA9"
    },
    "dark": {
      "bg": "#28456C",
      "fg": "#529CCA"
    },
    "name": "blue"
  },
  "sorting": {
    "light": {
      "bg": "#FBF3DB",
      "fg": "#CB912F"
    },
    "dark": {
      "bg": "#59563B",
      "fg": "#FFDC49"
    },
    "name": "yellow"
  },
  "gesture": {
    "light": {
      "bg": "#FBECDD",
      "fg": "#D9730D"
    },
    "dark": {
      "bg": "#594A36",
      "fg": "#FFA344"
    },
    "name": "orange"
  },
  "graphs": {
    "light": {
      "bg": "#EDF3EC",
      "fg": "#448361"
    },
    "dark": {
      "bg": "#354C4B",
      "fg": "#4DAB9A"
    },
    "name": "green"
  },
  "icmi": {
    "light": {
      "bg": "#EDEDEB",
      "fg": "#6B6B67"
    },
    "dark": {
      "bg": "#373737",
      "fg": "#B7B7B5"
    },
    "name": "gray"
  },
  "linux": {
    "light": {
      "bg": "#EDEDEB",
      "fg": "#6B6B67"
    },
    "dark": {
      "bg": "#373737",
      "fg": "#B7B7B5"
    },
    "name": "gray"
  },
  "strongly": {
    "light": {
      "bg": "#EDF3EC",
      "fg": "#448361"
    },
    "dark": {
      "bg": "#354C4B",
      "fg": "#4DAB9A"
    },
    "name": "green"
  },
  "video-backbone": {
    "light": {
      "bg": "#EDEDEB",
      "fg": "#6B6B67"
    },
    "dark": {
      "bg": "#373737",
      "fg": "#B7B7B5"
    },
    "name": "gray"
  },
  "aigc": {
    "light": {
      "bg": "#F6F3F9",
      "fg": "#9065B0"
    },
    "dark": {
      "bg": "#44355B",
      "fg": "#9A6DD7"
    },
    "name": "purple"
  },
  "egocentric": {
    "light": {
      "bg": "#EDEDEB",
      "fg": "#6B6B67"
    },
    "dark": {
      "bg": "#373737",
      "fg": "#B7B7B5"
    },
    "name": "gray"
  },
  "finetuning": {
    "light": {
      "bg": "#F6F3F9",
      "fg": "#9065B0"
    },
    "dark": {
      "bg": "#44355B",
      "fg": "#9A6DD7"
    },
    "name": "purple"
  },
  "foundational": {
    "light": {
      "bg": "#EDF3EC",
      "fg": "#448361"
    },
    "dark": {
      "bg": "#354C4B",
      "fg": "#4DAB9A"
    },
    "name": "green"
  },
  "icassp": {
    "light": {
      "bg": "#F6F3F9",
      "fg": "#9065B0"
    },
    "dark": {
      "bg": "#44355B",
      "fg": "#9A6DD7"
    },
    "name": "purple"
  },
  "object-detection": {
    "light": {
      "bg": "#EDF3EC",
      "fg": "#448361"
    },
    "dark": {
      "bg": "#354C4B",
      "fg": "#4DAB9A"
    },
    "name": "green"
  },
  "survey": {
    "light": {
      "bg": "#F4EEEE",
      "fg": "#9F6B53"
    },
    "dark": {
      "bg": "#493A34",
      "fg": "#D4A181"
    },
    "name": "brown"
  },
  "weak-supervision": {
    "light": {
      "bg": "#FBF3DB",
      "fg": "#CB912F"
    },
    "dark": {
      "bg": "#59563B",
      "fg": "#FFDC49"
    },
    "name": "yellow"
  },
  "aaai": {
    "light": {
      "bg": "#F6F3F9",
      "fg": "#9065B0"
    },
    "dark": {
      "bg": "#44355B",
      "fg": "#9A6DD7"
    },
    "name": "purple"
  },
  "architecture": {
    "light": {
      "bg": "#EDEDEB",
      "fg": "#6B6B67"
    },
    "dark": {
      "bg": "#373737",
      "fg": "#B7B7B5"
    },
    "name": "gray"
  },
  "challenges": {
    "light": {
      "bg": "#F1F1EF",
      "fg": "#37352F"
    },
    "dark": {
      "bg": "#2F2F2C",
      "fg": "#D4D4D4"
    },
    "name": "default"
  },
  "distress": {
    "light": {
      "bg": "#FDEBEC",
      "fg": "#D44C47"
    },
    "dark": {
      "bg": "#594141",
      "fg": "#FF7369"
    },
    "name": "red"
  },
  "fundamentals": {
    "light": {
      "bg": "#EDF3EC",
      "fg": "#448361"
    },
    "dark": {
      "bg": "#354C4B",
      "fg": "#4DAB9A"
    },
    "name": "green"
  },
  "geometry": {
    "light": {
      "bg": "#EDF3EC",
      "fg": "#448361"
    },
    "dark": {
      "bg": "#354C4B",
      "fg": "#4DAB9A"
    },
    "name": "green"
  },
  "iclr": {
    "light": {
      "bg": "#F6F3F9",
      "fg": "#9065B0"
    },
    "dark": {
      "bg": "#44355B",
      "fg": "#9A6DD7"
    },
    "name": "purple"
  },
  "interspeech": {
    "light": {
      "bg": "#F6F3F9",
      "fg": "#9065B0"
    },
    "dark": {
      "bg": "#44355B",
      "fg": "#9A6DD7"
    },
    "name": "purple"
  },
  "neurips": {
    "light": {
      "bg": "#F6F3F9",
      "fg": "#9065B0"
    },
    "dark": {
      "bg": "#44355B",
      "fg": "#9A6DD7"
    },
    "name": "purple"
  },
  "self-touch": {
    "light": {
      "bg": "#EDF3EC",
      "fg": "#448361"
    },
    "dark": {
      "bg": "#354C4B",
      "fg": "#4DAB9A"
    },
    "name": "green"
  },
  "skeleton": {
    "light": {
      "bg": "#EDF3EC",
      "fg": "#448361"
    },
    "dark": {
      "bg": "#354C4B",
      "fg": "#4DAB9A"
    },
    "name": "green"
  },
  "wacv-iccv": {
    "light": {
      "bg": "#E7F3F8",
      "fg": "#337EA9"
    },
    "dark": {
      "bg": "#28456C",
      "fg": "#529CCA"
    },
    "name": "blue"
  },
  "advanced": {
    "light": {
      "bg": "#F6F3F9",
      "fg": "#9065B0"
    },
    "dark": {
      "bg": "#44355B",
      "fg": "#9A6DD7"
    },
    "name": "purple"
  },
  "advanced-lecture": {
    "light": {
      "bg": "#F6F3F9",
      "fg": "#9065B0"
    },
    "dark": {
      "bg": "#44355B",
      "fg": "#9A6DD7"
    },
    "name": "purple"
  },
  "foundations-of-learning": {
    "light": {
      "bg": "#EDF3EC",
      "fg": "#448361"
    },
    "dark": {
      "bg": "#354C4B",
      "fg": "#4DAB9A"
    },
    "name": "green"
  },
  "generative-and-representation": {
    "light": {
      "bg": "#F6F3F9",
      "fg": "#9065B0"
    },
    "dark": {
      "bg": "#44355B",
      "fg": "#9A6DD7"
    },
    "name": "purple"
  },
  "languages": {
    "light": {
      "bg": "#F6F3F9",
      "fg": "#9065B0"
    },
    "dark": {
      "bg": "#44355B",
      "fg": "#9A6DD7"
    },
    "name": "purple"
  },
  "linear-filters": {
    "light": {
      "bg": "#E7F3F8",
      "fg": "#337EA9"
    },
    "dark": {
      "bg": "#28456C",
      "fg": "#529CCA"
    },
    "name": "blue"
  },
  "lrec": {
    "light": {
      "bg": "#EDEDEB",
      "fg": "#6B6B67"
    },
    "dark": {
      "bg": "#373737",
      "fg": "#B7B7B5"
    },
    "name": "gray"
  },
  "motion": {
    "light": {
      "bg": "#E7F3F8",
      "fg": "#337EA9"
    },
    "dark": {
      "bg": "#28456C",
      "fg": "#529CCA"
    },
    "name": "blue"
  },
  "plos-one": {
    "light": {
      "bg": "#EDEDEB",
      "fg": "#6B6B67"
    },
    "dark": {
      "bg": "#373737",
      "fg": "#B7B7B5"
    },
    "name": "gray"
  },
  "repair": {
    "light": {
      "bg": "#F9EEF3",
      "fg": "#C14C8A"
    },
    "dark": {
      "bg": "#533B4C",
      "fg": "#E255A1"
    },
    "name": "pink"
  },
  "resnet": {
    "light": {
      "bg": "#F6F3F9",
      "fg": "#9065B0"
    },
    "dark": {
      "bg": "#44355B",
      "fg": "#9A6DD7"
    },
    "name": "purple"
  },
  "sampling-and-multiscale": {
    "light": {
      "bg": "#E7F3F8",
      "fg": "#337EA9"
    },
    "dark": {
      "bg": "#28456C",
      "fg": "#529CCA"
    },
    "name": "blue"
  },
  "searching": {
    "light": {
      "bg": "#E7F3F8",
      "fg": "#337EA9"
    },
    "dark": {
      "bg": "#28456C",
      "fg": "#529CCA"
    },
    "name": "blue"
  },
  "sft": {
    "light": {
      "bg": "#F6F3F9",
      "fg": "#9065B0"
    },
    "dark": {
      "bg": "#44355B",
      "fg": "#9A6DD7"
    },
    "name": "purple"
  },
  "shortest-path": {
    "light": {
      "bg": "#EDF3EC",
      "fg": "#448361"
    },
    "dark": {
      "bg": "#354C4B",
      "fg": "#4DAB9A"
    },
    "name": "green"
  },
  "acii": {
    "light": {
      "bg": "#EDEDEB",
      "fg": "#6B6B67"
    },
    "dark": {
      "bg": "#373737",
      "fg": "#B7B7B5"
    },
    "name": "gray"
  },
  "acl-hlt": {
    "light": {
      "bg": "#F6F3F9",
      "fg": "#9065B0"
    },
    "dark": {
      "bg": "#44355B",
      "fg": "#9A6DD7"
    },
    "name": "purple"
  },
  "acm-mm": {
    "light": {
      "bg": "#EDEDEB",
      "fg": "#6B6B67"
    },
    "dark": {
      "bg": "#373737",
      "fg": "#B7B7B5"
    },
    "name": "gray"
  },
  "acm-tiis": {
    "light": {
      "bg": "#EDEDEB",
      "fg": "#6B6B67"
    },
    "dark": {
      "bg": "#373737",
      "fg": "#B7B7B5"
    },
    "name": "gray"
  },
  "augmented-humans": {
    "light": {
      "bg": "#EDEDEB",
      "fg": "#6B6B67"
    },
    "dark": {
      "bg": "#373737",
      "fg": "#B7B7B5"
    },
    "name": "gray"
  },
  "backtracking": {
    "light": {
      "bg": "#F6F3F9",
      "fg": "#9065B0"
    },
    "dark": {
      "bg": "#44355B",
      "fg": "#9A6DD7"
    },
    "name": "purple"
  },
  "bfs": {
    "light": {
      "bg": "#EDF3EC",
      "fg": "#448361"
    },
    "dark": {
      "bg": "#354C4B",
      "fg": "#4DAB9A"
    },
    "name": "green"
  },
  "binary-search": {
    "light": {
      "bg": "#E7F3F8",
      "fg": "#337EA9"
    },
    "dark": {
      "bg": "#28456C",
      "fg": "#529CCA"
    },
    "name": "blue"
  },
  "chi": {
    "light": {
      "bg": "#EDEDEB",
      "fg": "#6B6B67"
    },
    "dark": {
      "bg": "#373737",
      "fg": "#B7B7B5"
    },
    "name": "gray"
  },
  "cogn-affect-behav-neurosci": {
    "light": {
      "bg": "#EDEDEB",
      "fg": "#6B6B67"
    },
    "dark": {
      "bg": "#373737",
      "fg": "#B7B7B5"
    },
    "name": "gray"
  },
  "complexity": {
    "light": {
      "bg": "#F1F1EF",
      "fg": "#37352F"
    },
    "dark": {
      "bg": "#2F2F2C",
      "fg": "#D4D4D4"
    },
    "name": "default"
  },
  "computer-speech-language": {
    "light": {
      "bg": "#EDEDEB",
      "fg": "#6B6B67"
    },
    "dark": {
      "bg": "#373737",
      "fg": "#B7B7B5"
    },
    "name": "gray"
  },
  "dfs": {
    "light": {
      "bg": "#EDF3EC",
      "fg": "#448361"
    },
    "dark": {
      "bg": "#354C4B",
      "fg": "#4DAB9A"
    },
    "name": "green"
  },
  "discourse": {
    "light": {
      "bg": "#F1F1EF",
      "fg": "#37352F"
    },
    "dark": {
      "bg": "#2F2F2C",
      "fg": "#D4D4D4"
    },
    "name": "default"
  },
  "diss": {
    "light": {
      "bg": "#EDEDEB",
      "fg": "#6B6B67"
    },
    "dark": {
      "bg": "#373737",
      "fg": "#B7B7B5"
    },
    "name": "gray"
  },
  "divide-and-conquer": {
    "light": {
      "bg": "#FBECDD",
      "fg": "#D9730D"
    },
    "dark": {
      "bg": "#594A36",
      "fg": "#FFA344"
    },
    "name": "orange"
  },
  "eccv": {
    "light": {
      "bg": "#E7F3F8",
      "fg": "#337EA9"
    },
    "dark": {
      "bg": "#28456C",
      "fg": "#529CCA"
    },
    "name": "blue"
  },
  "emnlp": {
    "light": {
      "bg": "#F6F3F9",
      "fg": "#9065B0"
    },
    "dark": {
      "bg": "#44355B",
      "fg": "#9A6DD7"
    },
    "name": "purple"
  },
  "findings-of-acl": {
    "light": {
      "bg": "#F6F3F9",
      "fg": "#9065B0"
    },
    "dark": {
      "bg": "#44355B",
      "fg": "#9A6DD7"
    },
    "name": "purple"
  },
  "findings-of-emnlp": {
    "light": {
      "bg": "#F6F3F9",
      "fg": "#9065B0"
    },
    "dark": {
      "bg": "#44355B",
      "fg": "#9A6DD7"
    },
    "name": "purple"
  },
  "foundations": {
    "light": {
      "bg": "#EDF3EC",
      "fg": "#448361"
    },
    "dark": {
      "bg": "#354C4B",
      "fg": "#4DAB9A"
    },
    "name": "green"
  },
  "graph": {
    "light": {
      "bg": "#EDF3EC",
      "fg": "#448361"
    },
    "dark": {
      "bg": "#354C4B",
      "fg": "#4DAB9A"
    },
    "name": "green"
  },
  "greedy": {
    "light": {
      "bg": "#FBF3DB",
      "fg": "#CB912F"
    },
    "dark": {
      "bg": "#59563B",
      "fg": "#FFDC49"
    },
    "name": "yellow"
  },
  "hash-table": {
    "light": {
      "bg": "#E7F3F8",
      "fg": "#337EA9"
    },
    "dark": {
      "bg": "#28456C",
      "fg": "#529CCA"
    },
    "name": "blue"
  },
  "heap": {
    "light": {
      "bg": "#FBECDD",
      "fg": "#D9730D"
    },
    "dark": {
      "bg": "#594A36",
      "fg": "#FFA344"
    },
    "name": "orange"
  },
  "ieee-acm-taslp": {
    "light": {
      "bg": "#EDEDEB",
      "fg": "#6B6B67"
    },
    "dark": {
      "bg": "#373737",
      "fg": "#B7B7B5"
    },
    "name": "gray"
  },
  "ieee-fg": {
    "light": {
      "bg": "#EDEDEB",
      "fg": "#6B6B67"
    },
    "dark": {
      "bg": "#373737",
      "fg": "#B7B7B5"
    },
    "name": "gray"
  },
  "ieee-trans-affective-computing": {
    "light": {
      "bg": "#EDEDEB",
      "fg": "#6B6B67"
    },
    "dark": {
      "bg": "#373737",
      "fg": "#B7B7B5"
    },
    "name": "gray"
  },
  "ieee-trans-ind-electron": {
    "light": {
      "bg": "#EDEDEB",
      "fg": "#6B6B67"
    },
    "dark": {
      "bg": "#373737",
      "fg": "#B7B7B5"
    },
    "name": "gray"
  },
  "ieee-trans-instrum-meas": {
    "light": {
      "bg": "#EDEDEB",
      "fg": "#6B6B67"
    },
    "dark": {
      "bg": "#373737",
      "fg": "#B7B7B5"
    },
    "name": "gray"
  },
  "ieee-trans-intell-transp-syst": {
    "light": {
      "bg": "#EDEDEB",
      "fg": "#6B6B67"
    },
    "dark": {
      "bg": "#373737",
      "fg": "#B7B7B5"
    },
    "name": "gray"
  },
  "ijcnlp": {
    "light": {
      "bg": "#EDEDEB",
      "fg": "#6B6B67"
    },
    "dark": {
      "bg": "#373737",
      "fg": "#B7B7B5"
    },
    "name": "gray"
  },
  "image-formation": {
    "light": {
      "bg": "#E7F3F8",
      "fg": "#337EA9"
    },
    "dark": {
      "bg": "#28456C",
      "fg": "#529CCA"
    },
    "name": "blue"
  },
  "image-processing": {
    "light": {
      "bg": "#E7F3F8",
      "fg": "#337EA9"
    },
    "dark": {
      "bg": "#28456C",
      "fg": "#529CCA"
    },
    "name": "blue"
  },
  "iui": {
    "light": {
      "bg": "#EDEDEB",
      "fg": "#6B6B67"
    },
    "dark": {
      "bg": "#373737",
      "fg": "#B7B7B5"
    },
    "name": "gray"
  },
  "iwsds": {
    "light": {
      "bg": "#F6F3F9",
      "fg": "#9065B0"
    },
    "dark": {
      "bg": "#44355B",
      "fg": "#9A6DD7"
    },
    "name": "purple"
  },
  "japanese": {
    "light": {
      "bg": "#F9EEF3",
      "fg": "#C14C8A"
    },
    "dark": {
      "bg": "#533B4C",
      "fg": "#E255A1"
    },
    "name": "pink"
  },
  "jslhr": {
    "light": {
      "bg": "#EDEDEB",
      "fg": "#6B6B67"
    },
    "dark": {
      "bg": "#373737",
      "fg": "#B7B7B5"
    },
    "name": "gray"
  },
  "language": {
    "light": {
      "bg": "#F6F3F9",
      "fg": "#9065B0"
    },
    "dark": {
      "bg": "#44355B",
      "fg": "#9A6DD7"
    },
    "name": "purple"
  },
  "linked-list": {
    "light": {
      "bg": "#EDF3EC",
      "fg": "#448361"
    },
    "dark": {
      "bg": "#354C4B",
      "fg": "#4DAB9A"
    },
    "name": "green"
  },
  "method": {
    "light": {
      "bg": "#E7F3F8",
      "fg": "#337EA9"
    },
    "dark": {
      "bg": "#28456C",
      "fg": "#529CCA"
    },
    "name": "blue"
  },
  "mlmi": {
    "light": {
      "bg": "#F1F1EF",
      "fg": "#37352F"
    },
    "dark": {
      "bg": "#2F2F2C",
      "fg": "#D4D4D4"
    },
    "name": "default"
  },
  "naacl": {
    "light": {
      "bg": "#F6F3F9",
      "fg": "#9065B0"
    },
    "dark": {
      "bg": "#44355B",
      "fg": "#9A6DD7"
    },
    "name": "purple"
  },
  "naacl-workshop": {
    "light": {
      "bg": "#EDEDEB",
      "fg": "#6B6B67"
    },
    "dark": {
      "bg": "#373737",
      "fg": "#B7B7B5"
    },
    "name": "gray"
  },
  "neural-architectures": {
    "light": {
      "bg": "#F6F3F9",
      "fg": "#9065B0"
    },
    "dark": {
      "bg": "#44355B",
      "fg": "#9A6DD7"
    },
    "name": "purple"
  },
  "neurocognitive": {
    "light": {
      "bg": "#FDEBEC",
      "fg": "#D44C47"
    },
    "dark": {
      "bg": "#594141",
      "fg": "#FF7369"
    },
    "name": "red"
  },
  "phil-trans-r-soc-b": {
    "light": {
      "bg": "#EDEDEB",
      "fg": "#6B6B67"
    },
    "dark": {
      "bg": "#373737",
      "fg": "#B7B7B5"
    },
    "name": "gray"
  },
  "prefix-sum": {
    "light": {
      "bg": "#E7F3F8",
      "fg": "#337EA9"
    },
    "dark": {
      "bg": "#28456C",
      "fg": "#529CCA"
    },
    "name": "blue"
  },
  "probabilistic-models": {
    "light": {
      "bg": "#F6F3F9",
      "fg": "#9065B0"
    },
    "dark": {
      "bg": "#44355B",
      "fg": "#9A6DD7"
    },
    "name": "purple"
  },
  "queue": {
    "light": {
      "bg": "#EDEDEB",
      "fg": "#6B6B67"
    },
    "dark": {
      "bg": "#373737",
      "fg": "#B7B7B5"
    },
    "name": "gray"
  },
  "recursion": {
    "light": {
      "bg": "#F6F3F9",
      "fg": "#9065B0"
    },
    "dark": {
      "bg": "#44355B",
      "fg": "#9A6DD7"
    },
    "name": "purple"
  },
  "research-and-closing": {
    "light": {
      "bg": "#F4EEEE",
      "fg": "#9F6B53"
    },
    "dark": {
      "bg": "#493A34",
      "fg": "#D4A181"
    },
    "name": "brown"
  },
  "scientific-reports": {
    "light": {
      "bg": "#EDEDEB",
      "fg": "#6B6B67"
    },
    "dark": {
      "bg": "#373737",
      "fg": "#B7B7B5"
    },
    "name": "gray"
  },
  "segment-tree": {
    "light": {
      "bg": "#EDF3EC",
      "fg": "#448361"
    },
    "dark": {
      "bg": "#354C4B",
      "fg": "#4DAB9A"
    },
    "name": "green"
  },
  "self-collected": {
    "light": {
      "bg": "#EDF3EC",
      "fg": "#448361"
    },
    "dark": {
      "bg": "#354C4B",
      "fg": "#4DAB9A"
    },
    "name": "green"
  },
  "semiotica": {
    "light": {
      "bg": "#EDEDEB",
      "fg": "#6B6B67"
    },
    "dark": {
      "bg": "#373737",
      "fg": "#B7B7B5"
    },
    "name": "gray"
  },
  "siggraph-asia": {
    "light": {
      "bg": "#EDEDEB",
      "fg": "#6B6B67"
    },
    "dark": {
      "bg": "#373737",
      "fg": "#B7B7B5"
    },
    "name": "gray"
  },
  "sliding-window": {
    "light": {
      "bg": "#E7F3F8",
      "fg": "#337EA9"
    },
    "dark": {
      "bg": "#28456C",
      "fg": "#529CCA"
    },
    "name": "blue"
  },
  "soc-sci-and-medicine": {
    "light": {
      "bg": "#EDEDEB",
      "fg": "#6B6B67"
    },
    "dark": {
      "bg": "#373737",
      "fg": "#B7B7B5"
    },
    "name": "gray"
  },
  "stack": {
    "light": {
      "bg": "#EDEDEB",
      "fg": "#6B6B67"
    },
    "dark": {
      "bg": "#373737",
      "fg": "#B7B7B5"
    },
    "name": "gray"
  },
  "synthese": {
    "light": {
      "bg": "#EDEDEB",
      "fg": "#6B6B67"
    },
    "dark": {
      "bg": "#373737",
      "fg": "#B7B7B5"
    },
    "name": "gray"
  },
  "tree": {
    "light": {
      "bg": "#EDF3EC",
      "fg": "#448361"
    },
    "dark": {
      "bg": "#354C4B",
      "fg": "#4DAB9A"
    },
    "name": "green"
  },
  "trie": {
    "light": {
      "bg": "#EDF3EC",
      "fg": "#448361"
    },
    "dark": {
      "bg": "#354C4B",
      "fg": "#4DAB9A"
    },
    "name": "green"
  },
  "union-find": {
    "light": {
      "bg": "#EDF3EC",
      "fg": "#448361"
    },
    "dark": {
      "bg": "#354C4B",
      "fg": "#4DAB9A"
    },
    "name": "green"
  },
  "vision-and-language": {
    "light": {
      "bg": "#F6F3F9",
      "fg": "#9065B0"
    },
    "dark": {
      "bg": "#44355B",
      "fg": "#9A6DD7"
    },
    "name": "purple"
  },
  "channels": {
    "light": {
      "bg": "#EDEDEB",
      "fg": "#6B6B67"
    },
    "dark": {
      "bg": "#373737",
      "fg": "#B7B7B5"
    },
    "name": "gray"
  },
  "pixels": {
    "light": {
      "bg": "#EDEDEB",
      "fg": "#6B6B67"
    },
    "dark": {
      "bg": "#373737",
      "fg": "#B7B7B5"
    },
    "name": "gray"
  },
  "thedress": {
    "light": {
      "bg": "#F6F3F9",
      "fg": "#9065B0"
    },
    "dark": {
      "bg": "#44355B",
      "fg": "#9A6DD7"
    },
    "name": "purple"
  }
};
const DEFAULT = {
  light: { bg: '#F1F1EF', fg: '#37352F' },
  dark: { bg: '#2F2F2C', fg: '#D4D4D4' }
};

const PILL_SELECTOR = [
  '.metadata-property .multi-select-pill',
  '.metadata-container .multi-select-pill',
  '.metadata-properties .multi-select-pill',
  '.bases-view .multi-select-pill',
  '.bases-metadata-value .multi-select-pill'
].join(',');

const STATUS_CANDIDATE_SELECTOR = [
  '.metadata-property',
  '.bases-view td',
  '.bases-view .bases-td',
  '.bases-view .bases-table-cell',
  '.bases-view [data-property-key]',
  '.bases-view [data-property]',
  '.bases-view [data-column-key]'
].join(',');

const STATUS_KEYS = new Set(['done', 'in-progress', 'not-started', 'to-review', 'to-solve']);

function normalizeTagText(text) {
  return String(text || '')
    .replace(/[×✕✖]\s*$/g, '')
    .replace(/^#/, '')
    .trim()
    .replace(/\s+/g, '-')
    .toLowerCase() || null;
}

function pillText(el) {
  const content = el.querySelector('.multi-select-pill-content, [class*="pill-content"]');
  return content ? content.textContent : el.firstChild?.textContent || el.textContent;
}

function collectPills(node, out) {
  if (!(node instanceof HTMLElement)) return;
  if (node.matches(PILL_SELECTOR)) out.add(node);
  node.querySelectorAll(PILL_SELECTOR).forEach(el => out.add(el));
}

function collectStatusCandidates(node, out) {
  if (!(node instanceof HTMLElement)) return;
  if (node.matches(STATUS_CANDIDATE_SELECTOR)) out.add(node);
  node.querySelectorAll(STATUS_CANDIDATE_SELECTOR).forEach(el => out.add(el));
}

function propertyKey(row) {
  const attr = row.dataset?.propertyKey || row.dataset?.property || row.dataset?.columnKey;
  if (attr) return String(attr).replace(/^note\./, '').trim();
  const input = row.querySelector('.metadata-property-key-input');
  if (input?.value) return input.value.trim();
  const key = row.querySelector('.metadata-property-key, [class*="property-key"]');
  return key?.textContent?.trim() || '';
}

function statusKey(text) {
  const clean = String(text || '')
    .replace(/[✅🟡⬜]/gu, '')
    .trim()
    .replace(/[_\s]+/g, '-')
    .toLowerCase();
  if (clean === 'in-progress') return 'in-progress';
  if (clean === 'not-started') return 'not-started';
  if (clean === 'to-review') return 'to-review';
  if (clean === 'to-solve') return 'to-solve';
  if (clean === 'done') return 'done';
  return null;
}

function candidateText(el) {
  if (el instanceof HTMLInputElement || el instanceof HTMLTextAreaElement) return el.value;
  return el.textContent;
}

module.exports = class NotionPropertyTagsPlugin extends Plugin {
  onload() {
    this.pending = new Set();
    this.pendingStatus = new Set();
    this.mode = this.getMode();
    this.queueFullScan = this.queueFullScan.bind(this);

    // Initial pass only. Subsequent work is limited to newly inserted pills.
    this.queueFullScan();

    this.observer = new MutationObserver(mutations => {
      for (const mutation of mutations) {
        for (const node of mutation.addedNodes) {
          collectPills(node, this.pending);
          collectStatusCandidates(node, this.pendingStatus);
        }
      }
      if (this.pending.size || this.pendingStatus.size) this.scheduleFlush();
    });
    this.observer.observe(document.body, { childList: true, subtree: true });

    // Theme changes need a one-time recolor, but ordinary class/style mutations do not.
    this.themeObserver = new MutationObserver(() => {
      const next = this.getMode();
      if (next !== this.mode) {
        this.mode = next;
        this.queueFullScan();
      }
    });
    this.themeObserver.observe(document.body, { attributes: true, attributeFilter: ['class'] });

    this.registerEvent(this.app.workspace.on('file-open', this.queueFullScan));
    this.registerEvent(this.app.workspace.on('active-leaf-change', this.queueFullScan));
    this.registerDomEvent(document, 'input', event => {
      const row = event.target instanceof HTMLElement ? event.target.closest('.metadata-property') : null;
      if (row) {
        this.pendingStatus.add(row);
        this.scheduleFlush();
      }
    });
  }

  onunload() {
    this.observer?.disconnect();
    this.themeObserver?.disconnect();
    if (this.raf) cancelAnimationFrame(this.raf);
    if (this.fullTimer) clearTimeout(this.fullTimer);
  }

  getMode() {
    return document.body.classList.contains('theme-dark') ? 'dark' : 'light';
  }

  queueFullScan() {
    clearTimeout(this.fullTimer);
    this.fullTimer = window.setTimeout(() => {
      document.querySelectorAll(PILL_SELECTOR).forEach(el => this.pending.add(el));
      document.querySelectorAll(STATUS_CANDIDATE_SELECTOR).forEach(el => this.pendingStatus.add(el));
      this.scheduleFlush();
    }, 120);
  }

  scheduleFlush() {
    if (this.raf) return;
    this.raf = requestAnimationFrame(() => {
      this.raf = null;
      const batch = [...this.pending];
      this.pending.clear();
      for (const el of batch) this.stylePill(el);
      const statusBatch = [...this.pendingStatus];
      this.pendingStatus.clear();
      for (const el of statusBatch) this.styleStatus(el);
    });
  }

  stylePill(el) {
    if (!el.isConnected) return;
    const tag = normalizeTagText(pillText(el));
    if (!tag) return;
    const mode = this.getMode();
    if (el.dataset.notionTag === tag && el.dataset.notionTagMode === mode) return;
    const pair = (COLOR_MAP[tag] || DEFAULT)[mode];
    el.dataset.notionTag = tag;
    el.dataset.notionTagMode = mode;
    el.classList.add('notion-property-tag');
    el.style.setProperty('--notion-tag-bg', pair.bg);
    el.style.setProperty('--notion-tag-fg', pair.fg);
  }

  styleStatus(el) {
    if (!el.isConnected) return;

    let target = el;
    if (el.matches('.metadata-property')) {
      if (propertyKey(el).toLowerCase() !== 'status') return;
      target = el.querySelector(
        '.metadata-property-value input, .metadata-property-value, [class*="property-value"] input, [class*="property-value"]'
      );
      if (!target) return;
    } else {
      const declaredKey = propertyKey(el);
      if (declaredKey && declaredKey.toLowerCase() !== 'status') return;

      // Prefer the smallest text-bearing child so a Base cell receives a chip,
      // rather than painting an entire row or table container.
      const descendants = [...el.querySelectorAll('span, div, input')]
        .filter(node => STATUS_KEYS.has(statusKey(candidateText(node))));
      if (descendants.length) target = descendants[descendants.length - 1];
    }

    const key = statusKey(candidateText(target));
    if (!key) {
      target.classList?.remove('notion-property-status');
      target.style?.removeProperty('--notion-status-bg');
      target.style?.removeProperty('--notion-status-fg');
      return;
    }

    const mode = this.getMode();
    const pair = (COLOR_MAP[key] || DEFAULT)[mode];
    target.dataset.notionStatus = key;
    target.dataset.notionStatusMode = mode;
    target.classList.add('notion-property-status');
    target.style.setProperty('--notion-status-bg', pair.bg);
    target.style.setProperty('--notion-status-fg', pair.fg);
  }
};
