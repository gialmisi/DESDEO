/**
 * Translations for the cats and dogs demo.
 *
 * The demo is shown at science outreach events in Finland, so it can be
 * switched between English and Finnish. Only the demo itself is translated;
 * the rest of the web UI, the login page included, stays in English.
 *
 * The web UI has no translation library, and pulling one in for one page would
 * be heavy handed, so the strings are kept in a plain record here.
 *
 * NOTE: the Finnish strings have not been checked by a native speaker.
 */

import type { Animal } from './breeds';

export type Language = 'en' | 'fi';

export const LANGUAGES: Language[] = ['en', 'fi'];

/** What each language calls itself, for the language toggle. */
export const LANGUAGE_NAMES: Record<Language, string> = {
	en: 'English',
	fi: 'Suomi'
};

export type Strings = {
	/** Choosing an animal. */
	introTitle: string;
	introBody: string;
	introPrompt: string;
	chooseCats: string;
	chooseDogs: string;

	/** Looking for breeds. */
	explorerTitle: Record<Animal, string>;
	round: (n: number) => string;
	startOver: string;
	chooseThis: string;
	emptyTitle: Record<Animal, string>;
	emptyBody: string;
	suggestedBreeds: string;
	showAllTraits: string;
	hiddenTraitsNote: (n: number) => string;
	/** Candidates are numbered while exploring, so the breed stays a surprise. */
	candidate: (n: number) => string;
	/** Short direction markers shown next to a trait name. */
	max: string;
	min: string;
	noBreedsYet: string;
	searching: Record<Animal, string>;

	/** The chosen breed. */
	resultTitle: Record<Animal, string>;
	foundAfter: (n: number) => string;
	trait: string;
	value: string;
	disclaimer: string;
	keepLooking: string;
	pictured: (breed: string) => string;
	photographBy: (author: string, license: string) => string;
	viaCommons: string;

	/** Things that went wrong. */
	errorTitle: string;
	problemMissing: (animal: Animal) => string;
	noBreedsFound: string;

	/** Labels handed to the shared visualizations panel. */
	visualization: {
		title: string;
		noSolutions: string;
	};

	/** Labels handed to the shared preferences sidebar. */
	sidebar: {
		header: string;
		preferenceType: string;
		value: string;
		max: string;
		min: string;
		iterate: string;
	};
};

const EN: Strings = {
	introTitle: 'Find your ideal cat or dog breed',
	introBody:
		'Multiobjective optimization problems turn up in the unlikeliest places. The data behind this ' +
		'demo comes from two real survey studies on the behaviour of cat and dog breeds in Finland, but ' +
		'the problem formulation should not be taken too seriously. Whatever the method suggests, you ' +
		'are the decision maker and you make the final call.',
	introPrompt: 'I like...',
	chooseCats: 'Cats 😻',
	chooseDogs: 'Dogs 🐕',

	explorerTitle: { cat: 'Your ideal cat 😺', dog: 'Your ideal dog 🐶' },
	round: (n) => `round ${n}`,
	startOver: 'Start over',
	chooseThis: 'This is the one',
	emptyTitle: {
		cat: 'Describe the cat you would like',
		dog: 'Describe the dog you would like'
	},
	emptyBody:
		'Set each trait on the left to the value you wish for, then press Iterate. You will get a ' +
		'handful of candidates that come as close to your wishes as the survey data allows. Which ' +
		'breed each one is stays hidden until you pick your favourite.',
	suggestedBreeds: 'Candidates',
	showAllTraits: 'All traits',
	hiddenTraitsNote: (n) =>
		`${n} further ${n === 1 ? 'trait is' : 'traits are'} left out of this view and kept at a middle value. ` +
		'Switch on "All traits" to set them yourself.',
	candidate: (n) => `Candidate ${n}`,
	max: 'max',
	min: 'min',
	noBreedsYet: 'No candidates yet. Set your wishes and press Iterate.',
	searching: { cat: 'Looking for cats...', dog: 'Looking for dogs...' },

	resultTitle: { cat: 'Your ideal cat breed is', dog: 'Your ideal dog breed is' },
	foundAfter: (n) => `Found after ${n} ${n === 1 ? 'round' : 'rounds'} of the reference point method.`,
	trait: 'Trait',
	value: 'Value',
	disclaimer:
		'Remember that this is a playful example. A real breed choice deserves rather more thought ' +
		'than seven numbers from a survey.',
	keepLooking: 'Keep looking',
	pictured: (breed) => `Pictured: ${breed}.`,
	photographBy: (author, license) => `Photograph by ${author}, ${license},`,
	viaCommons: 'via Wikimedia Commons',

	errorTitle: 'Something went wrong',
	problemMissing: (animal) =>
		`The ${animal} breed problem was not found. Seed it with desdeo/api/db_init_catsanddogs.py.`,
	noBreedsFound: 'No breeds were found for this wish. Try changing it a little.',

	visualization: {
		title: 'How the candidates compare',
		noSolutions: 'No candidates to show yet.'
	},

	sidebar: {
		header: 'What would you like?',
		preferenceType: 'Your wishes',
		value: 'Value',
		max: 'max',
		min: 'min',
		iterate: 'Iterate'
	}
};

const FI: Strings = {
	introTitle: 'Löydä ihanteellinen kissa- tai koirarotusi',
	introBody:
		'Monitavoitteisia optimointiongelmia löytyy mitä yllättävimmistä paikoista. Tämän esittelyn ' +
		'aineisto on peräisin kahdesta todellisesta kyselytutkimuksesta, jotka käsittelivät kissa- ja ' +
		'koirarotujen käyttäytymistä Suomessa. Ongelman muotoilua ei kuitenkaan kannata ottaa liian ' +
		'vakavasti. Ehdottipa menetelmä mitä tahansa, sinä olet päätöksentekijä ja teet lopullisen ' +
		'valinnan.',
	introPrompt: 'Pidän...',
	chooseCats: 'Kissoista 😻',
	chooseDogs: 'Koirista 🐕',

	explorerTitle: { cat: 'Ihanteellinen kissasi 😺', dog: 'Ihanteellinen koirasi 🐶' },
	round: (n) => `kierros ${n}`,
	startOver: 'Aloita alusta',
	chooseThis: 'Tämä on se oikea',
	emptyTitle: {
		cat: 'Kuvaile millaisen kissan haluaisit',
		dog: 'Kuvaile millaisen koiran haluaisit'
	},
	emptyBody:
		'Aseta jokainen ominaisuus vasemmalla toivomaasi arvoon ja paina Etsi. Saat muutaman ' +
		'vaihtoehdon, jotka vastaavat toiveitasi niin hyvin kuin kyselyaineisto sallii. Mikä rotu ' +
		'kukin on, paljastuu vasta kun valitset suosikkisi.',
	suggestedBreeds: 'Vaihtoehdot',
	showAllTraits: 'Kaikki ominaisuudet',
	hiddenTraitsNote: (n) =>
		`${n} muuta ominaisuutta ei näytetä tässä näkymässä, ja ne pidetään keskiarvossa. ` +
		'Ota "Kaikki ominaisuudet" käyttöön asettaaksesi ne itse.',
	candidate: (n) => `Vaihtoehto ${n}`,
	max: 'maks',
	min: 'min',
	noBreedsYet: 'Ei vielä vaihtoehtoja. Aseta toiveesi ja paina Etsi.',
	searching: { cat: 'Etsitään kissoja...', dog: 'Etsitään koiria...' },

	resultTitle: { cat: 'Ihanteellinen kissarotusi on', dog: 'Ihanteellinen koirarotusi on' },
	foundAfter: (n) => `Löytyi ${n} kierroksen jälkeen viitepistemenetelmällä.`,
	trait: 'Ominaisuus',
	value: 'Arvo',
	disclaimer:
		'Muista, että tämä on leikkimielinen esimerkki. Todellinen rotuvalinta ansaitsee hieman ' +
		'enemmän harkintaa kuin seitsemän kyselystä saatua lukua.',
	keepLooking: 'Jatka etsimistä',
	pictured: (breed) => `Kuvassa: ${breed}.`,
	photographBy: (author, license) => `Kuva: ${author}, ${license},`,
	viaCommons: 'Wikimedia Commonsin kautta',

	errorTitle: 'Jokin meni pieleen',
	problemMissing: (animal) =>
		`${animal === 'cat' ? 'Kissarotujen' : 'Koirarotujen'} ongelmaa ei löytynyt. ` +
		'Luo se skriptillä desdeo/api/db_init_catsanddogs.py.',
	noBreedsFound: 'Toiveellasi ei löytynyt rotuja. Kokeile muuttaa sitä hieman.',

	visualization: {
		title: 'Vaihtoehtojen vertailu',
		noSolutions: 'Ei vielä vaihtoehtoja näytettäväksi.'
	},

	sidebar: {
		header: 'Millaisen haluaisit?',
		preferenceType: 'Toiveesi',
		value: 'Arvo',
		max: 'maks',
		min: 'min',
		iterate: 'Etsi'
	}
};

export const TRANSLATIONS: Record<Language, Strings> = { en: EN, fi: FI };

/**
 * The names of the behavioural traits, by objective symbol.
 *
 * The problems themselves carry English names, and the demo shows these
 * instead so that both languages stay in step and the emoji can be kept.
 * An objective that is not listed here falls back to the name in the problem.
 */
export const OBJECTIVE_NAMES: Record<Language, Record<string, string>> = {
	en: {
		fearfulness: 'Fearfulness 🙀',
		human_aggression: 'Aggression towards people 😾',
		activity_playfulness: 'Activity and playfulness 🧶',
		cat_sociability: 'Sociability towards cats 😻',
		human_sociability: 'Sociability towards people 🤝',
		litterbox_issues: 'Litterbox issues 🚽',
		excessive_grooming: 'Excessive grooming 💈',

		insecurity_score: 'Insecurity 😨',
		training_focus_score: 'Trainability 🎓',
		activity_playfulness_score: 'Activity and playfulness 🎾',
		aggressiveness_dominance_score: 'Aggressiveness and dominance 😤',
		human_sociability_score: 'Sociability towards people 🤝',
		dog_sociability_score: 'Sociability towards dogs 🐶',
		perseverance_score: 'Perseverance 🦴'
	},
	fi: {
		fearfulness: 'Arkuus 🙀',
		human_aggression: 'Aggressiivisuus ihmisiä kohtaan 😾',
		activity_playfulness: 'Aktiivisuus ja leikkisyys 🧶',
		cat_sociability: 'Sosiaalisuus kissoja kohtaan 😻',
		human_sociability: 'Sosiaalisuus ihmisiä kohtaan 🤝',
		litterbox_issues: 'Hiekkalaatikkoongelmat 🚽',
		excessive_grooming: 'Liiallinen turkinhoito 💈',

		insecurity_score: 'Epävarmuus 😨',
		training_focus_score: 'Koulutettavuus 🎓',
		activity_playfulness_score: 'Aktiivisuus ja leikkisyys 🎾',
		aggressiveness_dominance_score: 'Aggressiivisuus ja dominanssi 😤',
		human_sociability_score: 'Sosiaalisuus ihmisiä kohtaan 🤝',
		dog_sociability_score: 'Sosiaalisuus koiria kohtaan 🐶',
		perseverance_score: 'Sinnikkyys 🦴'
	}
};

const LANGUAGE_STORAGE_KEY = 'catsanddogs-language';

/** Reads the language chosen last time, so a kiosk keeps the language it was left in. */
export function storedLanguage(): Language {
	if (typeof localStorage === 'undefined') return 'en';
	const stored = localStorage.getItem(LANGUAGE_STORAGE_KEY);
	return stored === 'fi' || stored === 'en' ? stored : 'en';
}

/** Remembers the chosen language. Failing to store it is not worth reporting. */
export function storeLanguage(language: Language): void {
	try {
		localStorage.setItem(LANGUAGE_STORAGE_KEY, language);
	} catch {
		// Storage can be unavailable, in which case the choice just is not remembered.
	}
}
